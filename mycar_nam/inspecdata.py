import os
import time
from donkeycar.parts.tub_v2 import Tub

# Xác định đường dẫn thư mục xe hiện tại
CAR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CAR_DIR, 'data')

def get_tub_path():
    '''
    Hàm này thông minh hơn:
    1. Kiểm tra xem thư mục 'data/' có phải là Tub luôn không (như hình bạn gửi).
    2. Nếu không, nó sẽ tìm thư mục con mới nhất trong 'data/'.
    '''
    # Trường hợp 1: data/ chứa trực tiếp manifest.json
    if os.path.exists(os.path.join(DATA_DIR, 'manifest.json')):
        return DATA_DIR
    
    # Trường hợp 2: Tìm thư mục con (tub_1, tub_2...)
    sub_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR) 
                if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # Lọc ra những thư mục có chứa manifest.json
    valid_tubs = [d for d in sub_dirs if os.path.exists(os.path.join(d, 'manifest.json'))]

    if not valid_tubs:
        return None
    
    # Trả về tub mới nhất
    return max(valid_tubs, key=os.path.getmtime)

def inspect():
    tub_path = get_tub_path()
    if not tub_path:
        print("❌ LỖI: Không tìm thấy dữ liệu Tub nào trong thư mục data!")
        return

    print(f"📂 Đang kiểm tra Tub tại: {tub_path}")
    
    try:
        # Load Tub V2
        tub = Tub(tub_path, read_only=True)
        total_records = len(tub)
        print(f"📊 Tổng số records: {total_records}")
        
        if total_records == 0:
            print("⚠️ Tub rỗng, chưa có dữ liệu.")
            return

        print("-" * 40)
        print("🔍 Kiểm tra 10 record cuối cùng:")
        
        # Duyệt qua tub (Tub V2 là một iterator)
        count = 0
        for record in tub:
            count += 1
            # Chỉ in 10 dòng cuối
            if count < total_records - 10:
                continue
                
            idx = record['_index']
            # Lấy behavior (lưu ý key có thể thay đổi tùy config, thường là behavior/one_hot...)
            # Ta sẽ tìm key nào có chữ 'behavior'
            behavior_key = next((k for k in record.keys() if 'behavior/one_hot' in k), None)
            
            print(f"Record #{idx}:")
            
            if behavior_key:
                vec = record[behavior_key]
                state_name = "UNKNOWN"
                
                # Map vector sang tên trạng thái (Dựa theo route_logic.py của bạn)
                # Lưu ý: So sánh list float đôi khi cần sai số, nhưng ở đây ta so sánh chính xác cho đơn giản
                if vec == [1.0, 0.0, 0.0]: state_name = "NORMAL (Đi thẳng)"
                elif vec == [0.0, 1.0, 0.0]: state_name = "LEFT (Rẽ trái)"
                elif vec == [0.0, 0.0, 1.0]: state_name = "OBSTACLE (Né vật)"
                
                print(f"   ✅ Behavior: {vec} -> {state_name}")
            else:
                print("   ❌ CẢNH BÁO: Không tìm thấy dữ liệu Behavior trong record này!")
                
            # Kiểm tra góc lái để xem người lái có hoạt động không
            angle = record.get('user/angle', 0)
            throttle = record.get('user/throttle', 0)
            print(f"   🎮 Input: Angle={angle:.2f}, Throttle={throttle:.2f}")
            print("-" * 20)

    except Exception as e:
        print(f"❌ Lỗi khi đọc Tub: {e}")

if __name__ == "__main__":
    inspect()