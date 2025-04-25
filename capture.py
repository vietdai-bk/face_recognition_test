import cv2
import os
from datetime import datetime
import face_recognition
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TRAINING_DIR = "training_faces"

def collect_training_images(person_name, num_images=10, update_status=None):
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
        logger.debug(f"Đã tạo thư mục {TRAINING_DIR}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if update_status:
            update_status("Không thể truy cập camera!")
        logger.error("Không thể mở camera")
        return
    
    count = 0
    person_dir = os.path.join(TRAINING_DIR, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        logger.debug(f"Đã tạo thư mục {person_dir}")
    
    if update_status:
        update_status(f"Bắt đầu thu thập ảnh cho {person_name}. Nhấn 'q' để dừng sớm.")
    
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            if update_status:
                update_status("Không thể đọc frame từ camera!")
            logger.error("Không thể đọc frame từ camera")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(person_dir, f"{timestamp}_{count}.jpg")
            try:
                cv2.imwrite(img_path, frame)
                count += 1
                if update_status:
                    update_status(f"Đã lưu ảnh {count}/{num_images}")
                logger.debug(f"Đã lưu ảnh {img_path}")
            except Exception as e:
                if update_status:
                    update_status(f"Lỗi khi lưu ảnh: {str(e)}")
                logger.error(f"Lỗi khi lưu ảnh {img_path}: {str(e)}")
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Thu thập ảnh", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if update_status:
        update_status("Thu thập ảnh hoàn tất.")
    logger.debug("Đã hoàn tất thu thập ảnh")