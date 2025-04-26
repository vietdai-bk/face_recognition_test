import cv2
import os
from datetime import datetime
import face_recognition
import logging
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TRAINING_DIR = "training_faces"

def align_face(image, face_location):
    top, right, bottom, left = face_location
    face_image = image[top-40:bottom+40, left-40:right+40]
    return cv2.resize(face_image, (150, 150))

def collect_training_images(person_name, num_images=10, update_status=None, camera_id=0):
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
        logger.debug(f"Đã tạo thư mục {TRAINING_DIR}")
    
    cap = cv2.VideoCapture(camera_id)
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
    
    last_save_time = time.time()

    stable_face_count = 0
    face_locations_prev = []

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
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top

            if face_width < 50 or face_height < 50:
                if update_status:
                    update_status("Khuôn mặt quá nhỏ, di chuyển gần camera hơn.")
                logger.warning("Khuôn mặt quá nhỏ")
            else:
                if face_locations == face_locations_prev:
                    stable_face_count += 1
                else:
                    stable_face_count = 0

                if stable_face_count > 5:
                    if time.time() - last_save_time > 0.5:
                        aligned_face = align_face(frame, face_locations[0])

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(person_dir, f"{timestamp}_{count}.jpg")
                        try:
                            cv2.imwrite(img_path, aligned_face)
                            count += 1
                            last_save_time = time.time()

                            if update_status:
                                update_status(f"Đã lưu ảnh {count}/{num_images}")
                            logger.debug(f"Đã lưu ảnh {img_path}")
                        except Exception as e:
                            if update_status:
                                update_status(f"Lỗi khi lưu ảnh: {str(e)}")
                            logger.error(f"Lỗi khi lưu ảnh {img_path}: {str(e)}")
                else:
                    if update_status:
                        update_status("Đang ổn định khuôn mặt, vui lòng giữ nguyên vị trí.")
                    logger.info("Đang ổn định khuôn mặt...")

                face_locations_prev = face_locations
        
        elif len(face_locations) == 0:
            if update_status:
                update_status("Không phát hiện khuôn mặt.")
        else:
            if update_status:
                update_status("Phát hiện nhiều hơn 1 khuôn mặt. Vui lòng chỉ có 1 người.")
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Thu thập ảnh khuôn mặt", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    if update_status:
        update_status("Thu thập ảnh hoàn tất.")
    logger.debug("Đã hoàn tất thu thập ảnh")
