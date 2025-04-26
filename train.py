import face_recognition
import os
import pickle

TRAINING_DIR = "training_faces"
MODEL_PATH = "trained_face_model.pkl"

def load_existing_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        return model_data["encodings"], model_data["names"]
    else:
        return [], []

def train_face_recognition():
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)
        print(f"Đã tạo thư mục {TRAINING_DIR}")
    
    known_face_encodings, known_face_names = load_existing_model()

    for person_name in os.listdir(TRAINING_DIR):
        person_dir = os.path.join(TRAINING_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                image = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(image)

                if len(face_encodings) > 0:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
                else:
                    print(f"⚠️  Không tìm thấy khuôn mặt trong ảnh {img_name}. Bỏ qua.")
            except Exception as e:
                print(f"⚠️  Lỗi khi xử lý ảnh {img_name}: {str(e)}")

    model_data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Đã lưu mô hình huấn luyện vào {MODEL_PATH}")
    
    return model_data
