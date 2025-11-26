import cv2
import numpy as np
import time
import logging

class RouteManager:
    """
    Part này chịu trách nhiệm quản lý logic điều hướng phức tạp cho xe.
    - Phát hiện ngã ba (vạch đen ngang) để theo dõi tiến trình trên đường đua.
    - Phát hiện tín hiệu đặc biệt (như vạch màu xanh lá) để thực hiện hành động tức thời.
    - Sử dụng một State Machine để quyết định hành vi (Behavior) cần thiết tại mỗi thời điểm.
    - Kích hoạt các hành vi (như rẽ) trong một khoảng thời gian nhất định rồi tự động quay về trạng thái lái xe bình thường.
    """
    
    def __init__(self, cfg):
        """
        Khởi tạo RouteManager.
        :param cfg: Đối tượng config (myconfig.py) để lấy các tham số.
        """
        self.cfg = cfg
        
        # --- CẤU HÌNH KỊCH BẢN LỘ TRÌNH (STATE MACHINE) ---
        # Kịch bản này định nghĩa hành vi cần thực hiện SAU KHI đi qua mỗi ngã ba.
        # Ví dụ: state_sequence[1] là hành vi cần làm khi GẶP ngã ba thứ 2.
        self.state_sequence = [
            "Normal",     # Count 0 (Bắt đầu): Hành vi là Normal, khi gặp ngã 3 #1 vẫn giữ Normal.
            "Normal",     # Count 1 (Sau ngã 3 #1): Hành vi là Normal, chuẩn bị gặp ngã 3 #2.
            "Left_Turn",  # Count 2 (Sau ngã 3 #2): Hành vi là Rẽ Trái.
            "Left_Turn",  # Count 3 (Sau ngã 3 #3): Hành vi là Rẽ Trái.
            "Left_Turn",  # Count 4 (Sau ngã 3 #4): Hành vi là Rẽ Trái.
            "Normal",     # Count 5 (Sau Cầu): Hành vi là Normal.
            "Left_Turn"   # Count 6 (Ngã 3 cuối): Hành vi là Rẽ Trái.
        ]
        
        # Mapping tên behavior sang One-hot vector. THỨ TỰ PHẢI KHỚP VỚI myconfig.py
        # Ví dụ: BEHAVIOR_LIST = ['Normal', 'Left_Turn', 'Right_Turn']
        self.behavior_map = {
            "Normal":         [1.0, 0.0, 0.0],
            "Left_Turn":      [0.0, 1.0, 0.0],
            "Right_Turn":     [0.0, 0.0, 1.0]
        }
        
        # Trạng thái khởi tạo
        self.intersection_count = 0
        self.current_behavior = self.state_sequence[0]
        self.one_hot_state = self.behavior_map[self.current_behavior]
        
        # --- CẤU HÌNH THÔNG SỐ TỪ myconfig.py ---
        # Ngã ba
        self.ROI_TOP = getattr(cfg, 'ROI_TOP_INTERSECTION', 90)
        self.ROI_BOTTOM = getattr(cfg, 'ROI_BOTTOM_INTERSECTION', 120)
        self.THRESH_VAL = getattr(cfg, 'THRESH_VAL_INTERSECTION', 60)
        self.PIXEL_COUNT_TRIGGER = getattr(cfg, 'PIXEL_COUNT_TRIGGER_INTERSECTION', 1500)
        self.COOLDOWN_LIMIT = getattr(cfg, 'COOLDOWN_LIMIT_INTERSECTION', 100) # 5 giây ở 20Hz
        self.cooldown_counter = 0

        # Rẽ phải
        self.RIGHT_TURN_COOLDOWN_LIMIT = getattr(cfg, 'RIGHT_TURN_COOLDOWN_LIMIT', 30) # 1.5 giây ở 20Hz

        # Hành vi tạm thời (Action States)
        self.ACTION_DURATION = getattr(cfg, 'ACTION_DURATION', 0.5) # 0.5 giây
        self.action_state_active = False
        self.action_end_time = 0

        print("RouteManager with Timed Actions Initialized!")
        print(f"Route Plan: {self.state_sequence}")
        print(f"Action duration set to: {self.ACTION_DURATION}s")


    def set_manual_right(self):
        # --- SỬA: Xóa bỏ điều kiện kiểm tra mode, hoặc tự động tắt Auto ---
        self.is_auto_mode = False # Khi bấm nút, ép chuyển sang Manual luôn
        self.current_behavior = "Right_Turn"
        self.one_hot_state = self.behavior_map["Right_Turn"]
        print(">>> BUTTON PRESSED: FORCE Right_Turn (re phai)")

    def set_manual_left(self):
        # --- SỬA: Xóa bỏ điều kiện kiểm tra mode ---
        self.is_auto_mode = False # Khi bấm nút, ép chuyển sang Manual luôn
        self.current_behavior = "Left_Turn"
        self.one_hot_state = self.behavior_map["Left_Turn"]
        print(">>> BUTTON PRESSED: FORCE Left_Turn (Rẽ trái)")

    def detect_intersection(self, img_arr):
        """Phát hiện ngã ba bằng cách đếm pixel đen trong vùng ROI."""
        if img_arr is None: return False
        try:
            roi = img_arr[self.ROI_TOP:self.ROI_BOTTOM, :, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, self.THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
            pixel_count = cv2.countNonZero(thresh)
            return pixel_count > self.PIXEL_COUNT_TRIGGER
        except Exception as e:
            logging.error(f"Error in detect_intersection: {e}")
        return False

    def detect_right_turn_signal(self, img_arr):
        """Phát hiện tín hiệu rẽ phải (màu xanh lá) trong toàn bộ ảnh."""
        if img_arr is None: return False
        try:
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, self.cfg.RIGHT_TURN_COLOR_LOW, self.cfg.RIGHT_TURN_COLOR_HIGH)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > self.cfg.RIGHT_TURN_MIN_AREA:
                    return True
        except Exception as e:
            logging.error(f"Error in detect_right_turn_signal: {e}")
        return False

    def set_action_state(self, behavior_name):
        """Kích hoạt một trạng thái hành động (rẽ) và đặt bộ đếm thời gian."""
        if behavior_name in self.behavior_map and behavior_name != "Normal":
            logging.info(f"==> ACTION TRIGGERED: {behavior_name} for {self.ACTION_DURATION}s")
            self.current_behavior = behavior_name
            self.one_hot_state = self.behavior_map[behavior_name]
            self.action_state_active = True
            self.action_end_time = time.time() + self.ACTION_DURATION

    def detect_stop_sign(self, img_arr):
        """Phát hiện biển báo STOP màu đỏ."""
        if img_arr is None: return False
        try:
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv, self.STOP_SIGN_MIN_HSV1, self.STOP_SIGN_MAX_HSV1)
            mask2 = cv2.inRange(hsv, self.STOP_SIGN_MIN_HSV2, self.STOP_SIGN_MAX_HSV2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > self.STOP_SIGN_MIN_AREA:
                    return True
        except Exception as e:
            logging.error(f"Error in detect_stop_sign: {e}")
        return False

    def run(self, img_arr, user_mode):
        """
        Hàm chính chạy trong vòng lặp. Output giờ đây có thêm `should_stop`.
        """
        # --- ƯU TIÊN 0: XỬ LÝ BIỂN BÁO STOP (CHỈ KHI CHẠY AUTO) ---
        if user_mode != 'user':
            if not self.stop_triggered and self.detect_stop_sign(img_arr):
                logging.info("!!! STOP SIGN DETECTED - TRIGGERING STOP !!!")
                self.stop_triggered = True
            
            if self.stop_triggered:
                # Nếu đã kích hoạt dừng, trả về cờ dừng và thoát ngay lập tức
                return self.one_hot_state, self.current_behavior, True

        # --- ƯU TIÊN 1: HÀNH VI TẠM THỜI ---
        if self.action_state_active:
            if time.time() < self.action_end_time:
                return self.one_hot_state, self.current_behavior, False
            else:
                logging.info("==> Action finished. Resetting to Normal.")
                self.action_state_active = False
                self.current_behavior = "Normal"
                self.one_hot_state = self.behavior_map["Normal"]
                self.cooldown_counter = 0 

        # --- ƯU TIÊN 3: LỘ TRÌNH NGÃ BA ---
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif self.detect_intersection(img_arr):
            self.intersection_count += 1
            logging.info(f"!!! INTERSECTION DETECTED !!! Count: {self.intersection_count}")
            
            if self.intersection_count < len(self.state_sequence):
                new_behavior = self.state_sequence[self.intersection_count]
                if new_behavior != "Normal":
                    self.set_action_state(new_behavior)
                else:
                    self.current_behavior = new_behavior    
                    self.one_hot_state = self.behavior_map[new_behavior]
                    logging.info(f"==> ROUTE STATE: {new_behavior}")
            
            self.cooldown_counter = self.COOLDOWN_LIMIT

        return self.one_hot_state, self.current_behavior, False
