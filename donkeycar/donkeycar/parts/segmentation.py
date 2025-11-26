import cv2
import numpy as np
import time

class RoadSegmentation:
    '''
    Part xử lý ảnh chạy song song (Threaded).
    Nhiệm vụ:
    1. Segmentation: Tách làn đường khỏi nền.
    2. Calculate Area: Tính diện tích vùng đường đi được.
    '''
    def __init__(self, width=160, height=120, sensitivity=120, debug=False):
        self.width = width
        self.height = height
        self.sensitivity = sensitivity
        self.debug = debug
        
        # Kernel xử lý
        self.kernel = np.ones((3,3), np.uint8)
        self.kernel_aggressive = np.ones((9,9), np.uint8) # Cắt mạnh
        
        # Biến lưu trữ luồng
        self.img_in = None
        self.running = True
        self.road_area = 0          # Output quan trọng nhất
        self.mask_image = None      # Để debug (hiển thị lên web)

    def run(self, img_arr):
        # Hàm này dùng khi chạy non-threaded (test)
        self.img_in = img_arr
        self.update()
        return self.road_area, self.mask_image

    def update(self):
        # Hàm này chạy trên luồng riêng liên tục
        while self.running:
            if self.img_in is None:
                time.sleep(0.01)
                continue
                
            try:
                frame = self.img_in
                
                # 1. Resize & ROI
                frame_resized = cv2.resize(frame, (self.width, self.height))
                roi_height = int(self.height * 0.8) # Lấy 80% phía dưới
                roi = frame_resized[self.height - roi_height:, :]

                # 2. Xử lý ảnh (Gray -> Blur -> Threshold)
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # THRESH_BINARY_INV: Giả định đường trắng, vạch đen -> Vạch thành Trắng, Đường thành Đen
                _, mask_lines = cv2.threshold(blurred, self.sensitivity, 255, cv2.THRESH_BINARY_INV)

                # 3. Làm dày vạch (Hàn gắn vết đứt)
                mask_lines = cv2.erode(mask_lines, self.kernel, iterations=1)
                mask_lines = cv2.dilate(mask_lines, self.kernel, iterations=2)
                
                # Đóng nắp chai (chặn trên) để FloodFill không tràn ra ngoài
                cv2.line(mask_lines, (0, 0), (self.width, 0), 255, 2)

                # 4. Flood Fill (Tìm vùng đường)
                # mask_lines đang là: Vạch=255 (Trắng), Đường=0 (Đen)
                # FloodFill từ đáy giữa sẽ tô vùng Đường thành 255
                mask_filled = mask_lines.copy()
                h, w = mask_lines.shape
                seed_point = (w // 2, h - 1)
                
                # Chỉ fill nếu điểm xuất phát là màu đen (Đường)
                if mask_filled[seed_point[1], seed_point[0]] == 0:
                    cv2.floodFill(mask_filled, None, seed_point, 255)

                # 5. XOR: Lấy (Vùng đã Fill) loại bỏ (Vạch gốc) => Chỉ còn mặt đường
                mask_road_raw = cv2.bitwise_xor(mask_filled, mask_lines)

                # 6. Cắt gọt mạnh (Aggressive Erode)
                mask_road_cut = cv2.erode(mask_road_raw, self.kernel_aggressive, iterations=2)

                # 7. Lọc nhiễu: Chỉ giữ lại vùng lớn nhất (Main Road)
                contours, _ = cv2.findContours(mask_road_cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_final = np.zeros_like(mask_road_cut)
                
                current_area = 0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    current_area = cv2.contourArea(largest_contour)
                    
                    if current_area > 50:
                        cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)

                # 8. Cập nhật Output
                self.road_area = current_area # Diện tích mặt đường
                
                # Tạo ảnh debug (nếu cần xem trên web)
                if self.debug:
                    full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                    full_mask[self.height - roi_height:, :] = mask_final
                    self.mask_image = full_mask
                
            except Exception as e:
                print(f"Segmentation Error: {e}")
            
            # Nghỉ 1 chút để không chiếm hết CPU
            time.sleep(0.005)

    def run_threaded(self, img_arr):
        # Main loop chỉ việc gửi ảnh vào và lấy kết quả ra ngay lập tức
        self.img_in = img_arr
        # print('self.road_area',self.road_area)
        return self.road_area, self.mask_image

    def shutdown(self):
        self.running = False