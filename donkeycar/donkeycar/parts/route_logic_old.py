import cv2
import numpy as np
import logging

class RouteManager:
    '''
    Part xử lý logic đếm ngã ba và điều hướng hành vi cho xe.
    Input: Ảnh Camera (để detect vạch đen).
    Output: Vector hành vi (One-hot) để đưa vào Model AI.
    '''
    
    def __init__(self):
        # --- CẤU HÌNH TRẠNG THÁI (STATE MACHINE) ---
        # Kịch bản đề bài:
        # 0. Xuất phát: Đi thẳng (Normal)
        # 1. Ngã 3 thứ 1: Bỏ qua (Normal)
        # 2. Ngã 3 thứ 2: Quẹo (Left)
        # 3. Ngã 3 thứ 3: Quẹo (Left)
        # 4. Ngã 3 thứ 4: Quẹo (Left)
        # 5. Ngã 3 gặp Cầu: Đi thẳng qua cầu (Normal)
        # 6. Ngã 3 cuối cùng: Quẹo (Left)
        
        self.state_sequence = [
            "Normal",     # Count 0: Start -> Gặp ngã 3 #1
            "Normal",     # Count 1: Qua ngã 3 #1 -> Gặp ngã 3 #2
            "Left_Turn",  # Count 2: Qua ngã 3 #2 -> Gặp ngã 3 #3
            "Left_Turn",  # Count 3: Qua ngã 3 #3 -> Gặp ngã 3 #4
            "Left_Turn",  # Count 4: Qua ngã 3 #4 -> Gặp Cầu
            "Normal",     # Count 5: Qua Cầu      -> Gặp ngã 3 cuối
            "Left_Turn"   # Count 6: Ngã 3 cuối   -> Về đích
        ]
        
        # Mapping tên behavior sang One-hot vector
        # Thứ tự phải khớp với BEHAVIOR_LIST trong myconfig.py
        # [Normal, Left_Turn, Obstacle_Avoid]
        self.behavior_map = {
            "Normal":         [1.0, 0.0, 0.0],
            "Left_Turn":      [0.0, 1.0, 0.0],
            "Obstacle_Avoid": [0.0, 0.0, 1.0]
        }
        
        # --- TRẠNG THÁI HIỆN TẠI ---
        self.intersection_count = 0
        self.current_behavior = self.state_sequence[0]
        self.one_hot_state = self.behavior_map[self.current_behavior]
        
        # --- CẤU HÌNH MANUAL OVERRIDE (MỚI) ---
        self.is_auto_mode = True  # Mặc định là Tự động đếm
        print("RouteManager Initialized in AUTO MODE")

        # Cấu hình CV & Cooldown (Giữ nguyên)
        self.ROI_TOP = 90
        self.ROI_BOTTOM = 120
        self.THRESH_VAL = 60
        self.PIXEL_COUNT_TRIGGER = 300
        self.COOLDOWN_LIMIT = 100
        self.cooldown_counter = 0

    def toggle_mode(self):
        '''Chuyển đổi giữa Tự động và Thủ công'''
        self.is_auto_mode = not self.is_auto_mode
        mode_str = "AUTO (Computer Vision)" if self.is_auto_mode else "MANUAL (User Controller)"
        print(f"\n>>> MODE SWITCHED TO: {mode_str} <<<\n")

    def set_manual_normal(self):
        if not self.is_auto_mode:
            self.current_behavior = "Normal"
            self.one_hot_state = self.behavior_map["Normal"]
            print(">>> MANUAL SET: Normal (Đi thẳng)")

    def set_manual_left(self):
        if not self.is_auto_mode:
            self.current_behavior = "Left_Turn"
            self.one_hot_state = self.behavior_map["Left_Turn"]
            print(">>> MANUAL SET: Left_Turn (Rẽ trái)")

    def set_manual_obstacle(self):
        if not self.is_auto_mode:
            self.current_behavior = "Obstacle_Avoid"
            self.one_hot_state = self.behavior_map["Obstacle_Avoid"]
            print(">>> MANUAL SET: Obstacle_Avoid (Né vật)")

    # --- HÀM XỬ LÝ CHÍNH ---

    def detect_intersection(self, img_arr):
        # (Giữ nguyên code xử lý ảnh cũ)
        if img_arr is None: return False
        try:
            roi = img_arr[self.ROI_TOP:self.ROI_BOTTOM, :, :]
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, self.THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
            count = cv2.countNonZero(thresh)
            # print('black pixel count: ',count)
            if count > self.PIXEL_COUNT_TRIGGER: return True
        except: pass
        return False

    def run(self, img_arr):
        # 1. Nếu đang ở chế độ Manual, bỏ qua logic đếm, giữ nguyên state do người dùng bấm
        if not self.is_auto_mode:
            return self.one_hot_state, self.current_behavior

        # 2. Nếu đang ở Auto Mode, chạy logic đếm như cũ
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.one_hot_state, self.current_behavior

        if self.detect_intersection(img_arr):
            self.intersection_count += 1
            print(f"!!! AUTO DETECT: Intersection #{self.intersection_count}")
            
            if self.intersection_count < len(self.state_sequence):
                new_behavior = self.state_sequence[self.intersection_count]
                self.current_behavior = new_behavior
                self.one_hot_state = self.behavior_map[new_behavior]
                print(f"==> Auto Switch to: {new_behavior}")
            
            self.cooldown_counter = self.COOLDOWN_LIMIT

        return self.one_hot_state, self.current_behavior