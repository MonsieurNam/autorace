import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class DualLineFollower:
    def __init__(self, pid, cfg):
        self.cfg = cfg
        self.overlay_image = getattr(cfg, 'OVERLAY_IMAGE', False)
        
        self.scan_y_near = cfg.SCAN_Y
        self.scan_height = cfg.SCAN_HEIGHT
        self.scan_y_far = max(0, cfg.SCAN_Y - 40)

        self.steering = 0.0
        self.throttle = cfg.THROTTLE_INITIAL
        self.turn_state = "STRAIGHT"
        
        self.last_error = 0.0 
        self.last_steering = 0.0 

        self.pid_st = pid
        self.pid_st.setpoint = 0
        
        self.morph_kernel = np.ones((3,3), np.uint8)
        self.search_margin = 30
        self.last_left_x = None
        self.last_right_x = None
        
        # Logic Đếm Cua
        self.left_turn_counter = 0       
        self.is_in_turn = False          
        self.force_straight_mode = False 
        
        self.consecutive_curve_frames = 0  
        self.consecutive_straight_frames = 0 
        
        logger.info("DualLineFollower Phase 8 (Logic Split) initialized.")

    def _apply_roi_mask(self, gray_img):
        h, w = gray_img.shape
        margin_left = getattr(self.cfg, 'ROI_MASK_LEFT', 0)
        margin_right = getattr(self.cfg, 'ROI_MASK_RIGHT', 0)
        if margin_left > 0: gray_img[:, :margin_left] = 255
        if margin_right > 0: gray_img[:, w - margin_right:] = 255
        return gray_img

    def _process_mask_advanced(self, mask):
        mask = cv2.erode(mask, self.morph_kernel, iterations=1)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=2)
        return mask

    def _find_lines_in_slice(self, cam_img, scan_y):
        roi = cam_img[scan_y : scan_y + self.scan_height, :, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        gray = self._apply_roi_mask(gray)
        _, mask = cv2.threshold(gray, self.cfg.GRAYSCALE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        mask = self._process_mask_advanced(mask)
        
        hist = np.sum(mask, axis=0)
        width = len(hist)
        midpoint = width // 2
        
        left_part = hist[:midpoint]
        left_x = np.argmax(left_part)
        left_conf = left_part[left_x]
        
        right_part = hist[midpoint:]
        right_x = np.argmax(right_part) + midpoint
        right_conf = right_part[np.argmax(right_part)]
        
        found_left = left_conf > self.cfg.CONFIDENCE_THRESHOLD
        found_right = right_conf > self.cfg.CONFIDENCE_THRESHOLD
        
        center = self.cfg.CAR_CENTER_PIXEL
        if found_left and found_right:
            center = (left_x + right_x) // 2
        elif found_left:
            center = left_x + (self.cfg.LANE_WIDTH_PIXELS // 2)
        elif found_right:
            center = right_x - (self.cfg.LANE_WIDTH_PIXELS // 2)
            
        return center, found_left, found_right, mask, left_x, right_x

    def _calculate_dynamic_throttle(self, error, is_sharp_curve):
        th_min = getattr(self.cfg, 'THROTTLE_MIN', 0.15)
        th_max = getattr(self.cfg, 'THROTTLE_MAX', 0.25)
        th_step = getattr(self.cfg, 'THROTTLE_STEP', 0.02)
        brake_threshold = getattr(self.cfg, 'THROTTLE_BRAKE_THRESHOLD', 10)

        target_throttle = th_max
        if is_sharp_curve or abs(error) > brake_threshold:
            target_throttle = th_min
        
        if self.throttle < target_throttle: self.throttle += th_step
        elif self.throttle > target_throttle: self.throttle -= th_step * 2
            
        self.throttle = max(min(self.throttle, th_max), th_min)
        return self.throttle

    def run(self, cam_img):
        if cam_img is None: return 0.0, 0.0, None

        # 1. LẤY THÔNG TIN VẠCH (Dòng Gần)
        center_near, f_left, f_right, mask_near, left_x, right_x = self._find_lines_in_slice(cam_img, self.scan_y_near)
        
        # 2. LẤY THÔNG TIN VẠCH (Dòng Xa) - Để đếm cua
        center_far, _, _, mask_far, _, _ = self._find_lines_in_slice(cam_img, self.scan_y_far)
        lane_shift = center_far - center_near
        
        # --- LOGIC ĐẾM CUA (TURN COUNTING) ---
        CURVE_THRESHOLD = 25  
        current_status = "STRAIGHT"
        if lane_shift < -CURVE_THRESHOLD: current_status = "LEFT"
        elif lane_shift > CURVE_THRESHOLD: current_status = "RIGHT"

        # Lọc nhiễu
        if current_status == "LEFT":
            self.consecutive_curve_frames += 1
            self.consecutive_straight_frames = 0
        elif current_status == "STRAIGHT":
            self.consecutive_straight_frames += 1
            self.consecutive_curve_frames = 0
        else:
            self.consecutive_curve_frames = 0

        # Phát hiện vào cua
        if not self.is_in_turn and self.consecutive_curve_frames > 3:
            self.left_turn_counter += 1
            self.is_in_turn = True
            logger.info(f" >>> DETECTED LEFT TURN #{self.left_turn_counter}")

        # Phát hiện hết cua
        if self.is_in_turn and self.consecutive_straight_frames > 10:
            self.is_in_turn = False
            self.force_straight_mode = False
            logger.info(" >>> EXITED TURN")

        # --- QUYẾT ĐỊNH LOGIC LÁI ---
        
        use_custom_logic = False
        
        if self.is_in_turn:
            if self.left_turn_counter == 1:
                self.turn_state = "SKIP LEFT #1"
                use_custom_logic = True
                # self.force_straight_mode = True # Sẽ xử lý ở dưới
            elif self.left_turn_counter == 2:
                self.turn_state = "TAKE LEFT #2"
                use_custom_logic = True # BẬT CHẾ ĐỘ LÁI CỦA BẠN
        else:
            self.turn_state = "STRAIGHT"

        # Biến để tính error
        error = 0.0
        status_code = 0 
        lane_center = 0
        is_lost = False

        # ============================================================
        # NHÁNH 1: LOGIC TÙY CHỈNH CỦA BẠN (KHI RẼ TRÁI LẦN 2)
        # ============================================================
        if use_custom_logic:
            if f_left and f_right:
                lane_center = (left_x + right_x) // 2
                error = lane_center - self.cfg.CAR_CENTER_PIXEL
                status_code = 0
                # Cập nhật độ rộng
                current_width = right_x - left_x
                if 50 < current_width < 80:
                    self.cfg.LANE_WIDTH_PIXELS = int(0.95 * self.cfg.LANE_WIDTH_PIXELS + 0.05 * current_width)

            elif f_left and not f_right:
                # Logic ép trái mạnh
                if self.cfg.CAR_CENTER_PIXEL - left_x > 40: 
                    lane_center = left_x + (self.cfg.LANE_WIDTH_PIXELS // 2)
                    error = lane_center - self.cfg.CAR_CENTER_PIXEL
                    status_code = 1
                    is_lost = True 
                else:
                    # Hardcode về 0 để bẻ lái cực gắt
                    lane_center = 0 
                    error = lane_center - self.cfg.CAR_CENTER_PIXEL 
                    status_code = 1
                    is_lost = True

            elif not f_left and f_right:
                # Logic Hardcode bên phải
                lane_center = 0 # Ép về 0 luôn
                error = lane_center - self.cfg.CAR_CENTER_PIXEL
                status_code = 2
                is_lost = True

            else:
                error = self.last_error
                status_code = 3
                is_lost = True

        # ============================================================
        # NHÁNH 2: LOGIC TIÊU CHUẨN (ĐI THẲNG / SKIP CUA 1)
        # ============================================================
        else:
            if self.force_straight_mode:
                # Nếu đang Skip cua 1 -> Ép xe đi thẳng
                error = 0
                status_code = 0
                lane_center = self.cfg.CAR_CENTER_PIXEL
            
            else:
                # Logic bình thường (An toàn)
                if f_left and f_right:
                    lane_center = (left_x + right_x) // 2
                    error = lane_center - self.cfg.CAR_CENTER_PIXEL
                    status_code = 0
                elif f_left:
                    lane_center = left_x + (self.cfg.LANE_WIDTH_PIXELS // 2)
                    error = lane_center - self.cfg.CAR_CENTER_PIXEL
                    status_code = 1
                elif f_right:
                    lane_center = right_x - (self.cfg.LANE_WIDTH_PIXELS // 2)
                    error = lane_center - self.cfg.CAR_CENTER_PIXEL
                    status_code = 2
                else:
                    error = self.last_error
                    status_code = 3
                    is_lost = True

        # ============================================================

        self.last_error = error

        # --- PID CONTROL ---
        raw_steering = self.pid_st(error)
        smooth = getattr(self.cfg, 'STEERING_SMOOTH_FACTOR', 1.0)
        self.steering = (smooth * raw_steering) + ((1.0 - smooth) * self.last_steering)
        self.last_steering = self.steering

        # Tốc độ động
        is_sharp_curve = abs(lane_shift) > 25 or is_lost
        if self.force_straight_mode: is_sharp_curve = False 
        self.throttle = self._calculate_dynamic_throttle(error, is_sharp_curve)

        # --- HIỂN THỊ ---
        if self.overlay_image:
            overlay_img = self.overlay_display_dual(cam_img, lane_center, center_far, f_left, f_right, left_x, right_x, status_code)
            return self.steering, self.throttle, overlay_img
        
        return self.steering, self.throttle, cam_img

    def overlay_display_dual(self, cam_img, lane_center, center_far, f_left, f_right, left_x, right_x, status_code):
        img = np.copy(cam_img)
        h, w, _ = img.shape
        
        m_left = getattr(self.cfg, 'ROI_MASK_LEFT', 0)
        m_right = getattr(self.cfg, 'ROI_MASK_RIGHT', 0)
        if m_left > 0: img[:, :m_left] //= 2
        if m_right > 0: img[:, w - m_right:] //= 2

        y_near = self.scan_y_near + (self.scan_height // 2)
        y_far = self.scan_y_far + (self.scan_height // 2)
        
        if f_left: cv2.line(img, (left_x, y_near-5), (left_x, y_near+5), (255, 0, 0), 2)
        if f_right: cv2.line(img, (right_x, y_near-5), (right_x, y_near+5), (0, 0, 255), 2)
        cv2.circle(img, (lane_center, y_near), 3, (0, 255, 255), -1)

        pred_color = (0, 255, 0) 
        pred_text = "^ STR"
        
        if "SKIP" in self.turn_state:
            pred_color = (0, 0, 255) 
            pred_text = "^ SKIP"
        elif "TAKE" in self.turn_state:
            pred_color = (255, 200, 0) 
            pred_text = "< TURN"
        elif "IN TURN" in self.turn_state:
             pred_color = (0, 255, 255)
             pred_text = "Curve..."

        # Nếu đang ép rẽ, vẽ mũi tên chỉ sang trái
        target_far = center_far
        if self.force_straight_mode: target_far = lane_center
        
        cv2.arrowedLine(img, (lane_center, y_near), (target_far, y_far), pred_color, 2)
        
        # HUD Compact
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (255, 255, 255), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        font_scale = 0.35
        font_color = (0, 0, 0) 
        thickness = 1
        
        cv2.putText(img, f"{pred_text} | Cnt:{self.left_turn_counter}", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
        cv2.putText(img, f"St:{self.steering:.2f} Th:{self.throttle:.2f}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
        
        status_msg = "OK"
        if status_code == 1: status_msg = "NO R"
        elif status_code == 2: status_msg = "NO L"
        elif status_code == 3: status_msg = "LOST"
        
        text_size = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        cv2.putText(img, status_msg, (w - text_size[0] - 5, 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

        cv2.line(img, (self.cfg.CAR_CENTER_PIXEL, y_near-5), (self.cfg.CAR_CENTER_PIXEL, y_near+5), (100, 100, 100), 1)
        
        return img