import face_recognition
import os

TRAINING_DIR = "training_faces"

def train_face_recognition():
    # Kiểm tra và tạo thư mục training_faces nếu chưa tồn tại
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
    
    known_face_encodings = []
    known_face_names = []
    
    for person_name in os.listdir(TRAINING_DIR):
        person_dir = os.path.join(TRAINING_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
    
    return known_face_encodings, known_face_names