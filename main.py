import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import os
import pickle
from capture import collect_training_images
from train import train_face_recognition
from recognize import recognize_faces
import logging
from tkinter import filedialog
import pandas as pd

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = "trained_face_model.pkl"

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Điểm Danh Khuôn Mặt")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(True, True)

        self.known_face_encodings, self.known_face_names = self.load_trained_model()

        self.stop_event = threading.Event()
        self.recognition_thread = None
        self.attendance_status = {name: False for name in self.known_face_names}
        self.attendance_file = None

        self.main_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.create_widgets()

    def load_trained_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    model_data = pickle.load(f)
                print("✅ Đã load model thành công.")
                return model_data["encodings"], model_data["names"]
            except Exception as e:
                print(f"⚠️ Lỗi khi load model: {e}")
        else:
            print("⚠️ Không tìm thấy model, khởi tạo dữ liệu rỗng.")
        return [], []
    
    def open_face_list_window(self):
        face_list_window = tk.Toplevel(self.root)
        face_list_window.title("Danh Sách Khuôn Mặt")
        face_list_window.geometry("300x400")
        face_list_window.configure(bg="#f0f4f8")

        tk.Label(
            face_list_window,
            text="Danh sách khuôn mặt đã thêm:",
            font=("Helvetica", 14, "bold"),
            bg="#f0f4f8",
            fg="#2c3e50"
        ).pack(pady=10)

        listbox = tk.Listbox(
            face_list_window,
            font=("Helvetica", 12),
            bg="white",
            fg="#2c3e50",
            width=30,
            height=15
        )
        listbox.pack(pady=10)

        unique_names = list(set(self.known_face_names))
        unique_names.sort()

        for name in unique_names:
            listbox.insert(tk.END, name)

        def delete_face(event):
            selected_face = listbox.get(tk.ACTIVE)
            if not selected_face:
                return

            confirm = messagebox.askyesno(
                "Xóa khuôn mặt", 
                f"Bạn có chắc chắn muốn xóa khuôn mặt của {selected_face} khỏi mô hình?"
            )
            if confirm:
                self.delete_face_from_model(selected_face)
                listbox.delete(tk.ACTIVE)
                self.update_model()

        listbox.bind("<Button-3>", delete_face)

        tk.Button(
            face_list_window,
            text="Đóng",
            command=face_list_window.destroy,
            font=("Helvetica", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            width=10,
            height=1,
            relief="flat"
        ).pack(pady=10)
    
    def delete_face_from_model(self, face_name):
        if face_name in self.known_face_names:
            indices_to_delete = [i for i, name in enumerate(self.known_face_names) if name == face_name]
            for index in sorted(indices_to_delete, reverse=True):
                del self.known_face_names[index]
                del self.known_face_encodings[index]
            print(f"✅ Đã xóa khuôn mặt của {face_name} khỏi mô hình.")
            print("Danh sách tên khuôn mặt sau khi xóa:", self.known_face_names)
            print("Danh sách mã hóa khuôn mặt sau khi xóa:", self.known_face_encodings)
            self.update_model()
            self.face_count_label.config(text=self.get_face_count_text())
            self.update_attendance_status()
        else:
            print(f"⚠️ Không tìm thấy khuôn mặt {face_name} trong mô hình.")
    
    def update_model(self):
        model_data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names
        }
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
            print(f"✅ Đã lưu mô hình mới vào {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu mô hình: {str(e)}")

    def create_widgets(self):
        self.title_label = tk.Label(
            self.main_frame, 
            text="Hệ Thống Điểm Danh Khuôn Mặt",
            font=("Helvetica", 24, "bold"),
            bg="#f0f4f8",
            fg="#2c3e50"
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="n")

        self.video_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.video_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="n")
        self.video_label = tk.Label(
            self.video_frame,
            text="Chào mừng đến với hệ thống điểm danh!",
            font=("Helvetica", 14),
            bg="#d3dce6",
            fg="#34495e",
            width=80,
            height=30
        )
        self.video_label.pack()

        self.face_count_label = tk.Label(
            self.main_frame,
            text=self.get_face_count_text(),
            font=("Helvetica", 14),
            bg="#f0f4f8",
            fg="#2c3e50"
        )
        self.face_count_label.grid(row=2, column=0, columnspan=2, pady=10, sticky="n")

        self.attendance_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.attendance_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        self.attendance_label = tk.Label(
            self.attendance_frame,
            text="Trạng thái điểm danh: Chưa có ai",
            font=("Helvetica", 12),
            bg="#f0f4f8",
            fg="#2c3e50",
            wraplength=1000
        )
        self.attendance_label.pack(fill="x")

        self.status_label = tk.Label(
            self.main_frame,
            text="Sẵn sàng",
            font=("Helvetica", 12),
            bg="#f0f4f8",
            fg="#2c3e50"
        )
        self.status_label.grid(row=4, column=0, columnspan=2, pady=10, sticky="n")

        self.button_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.button_frame.grid(row=5, column=0, columnspan=2, pady=20, sticky="ew")

        self.inner_button_frame = tk.Frame(self.button_frame, bg="#f0f4f8")
        self.inner_button_frame.pack(expand=True)
        self.view_faces_button = tk.Button(
            self.inner_button_frame,
            text="Xem Danh Sách Khuôn Mặt",
            command=self.open_face_list_window,
            font=("Helvetica", 14, "bold"),
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            activeforeground="white",
            width=20,
            height=2,
            relief="flat"
        )
        self.view_faces_button.pack(side="left", padx=20, pady=10)


        self.add_face_button = tk.Button(
            self.inner_button_frame,
            text="Thêm Khuôn Mặt",
            command=self.open_add_face_window,
            font=("Helvetica", 14, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            width=20,
            height=2,
            relief="flat"
        )
        self.add_face_button.pack(side="left", padx=20, pady=10)

        self.start_button = tk.Button(
            self.inner_button_frame,
            text="Bắt Đầu Điểm Danh",
            command=self.start_recognition,
            font=("Helvetica", 14, "bold"),
            bg="#2ecc71",
            fg="white",
            activebackground="#27ae60",
            activeforeground="white",
            width=20,
            height=2,
            relief="flat"
        )
        self.start_button.pack(side="left", padx=20, pady=10)

        self.stop_button = tk.Button(
            self.inner_button_frame,
            text="Kết Thúc Điểm Danh",
            command=self.stop_recognition,
            font=("Helvetica", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            width=20,
            height=2,
            relief="flat",
            state=tk.DISABLED
        )
        self.stop_button.pack(side="left", padx=20, pady=10)

        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(5, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def get_face_count_text(self):
        unique_names = list(set(self.known_face_names))
        return f"Số khuôn mặt đã thêm: {len(unique_names)}"

    def update_status(self, message):
        self.status_label.config(text=message)

    def update_attendance_status(self):
        status_text = "Trạng thái điểm danh: " + ", ".join(
            f"{name}: {'Đã điểm danh' if status else 'Chưa điểm danh'}"
            for name, status in self.attendance_status.items()
        )
        self.attendance_label.config(text=status_text)

    def open_add_face_window(self):
        add_face_window = tk.Toplevel(self.root)
        add_face_window.title("Thêm Khuôn Mặt")
        add_face_window.geometry("300x200")
        add_face_window.configure(bg="#f0f4f8")

        tk.Label(
            add_face_window,
            text="Nhập tên:",
            font=("Helvetica", 12),
            bg="#f0f4f8",
            fg="#2c3e50"
        ).pack(pady=20)

        name_entry = tk.Entry(add_face_window, width=20, font=("Helvetica", 12))
        name_entry.pack(pady=10)
        name_entry.insert(0, "Nhập tên")

        def add_face():
            name = name_entry.get().strip()
            if not name or name == "Nhập tên":
                messagebox.showerror("Lỗi", "Vui lòng nhập tên hợp lệ!")
                return

            self.update_status("Đang thu thập ảnh...")
            collect_training_images(name, num_images=10, update_status=self.update_status)
            train_face_recognition()
            self.known_face_encodings, self.known_face_names = self.load_trained_model()
            self.attendance_status = {name: False for name in self.known_face_names}
            self.face_count_label.config(text=self.get_face_count_text())
            self.update_attendance_status()
            self.update_status("Đã thêm khuôn mặt và huấn luyện lại mô hình.")
            add_face_window.destroy()

        tk.Button(
            add_face_window,
            text="Xác nhận",
            command=add_face,
            font=("Helvetica", 12, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            width=10,
            height=1,
            relief="flat"
        ).pack(pady=20)
    
    def update_attendance_status(self):
        attended_names = [name for name, attended in self.attendance_status.items() if attended]
        
        if attended_names:
            attended_text = "Những người đã điểm danh: " + ", ".join(attended_names)
        else:
            attended_text = "Chưa có ai điểm danh."
        self.attendance_label.config(text=attended_text)

    def start_recognition(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file Excel để lưu danh sách điểm danh",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not file_path:
            return
        self.attendance_file = file_path
        if os.path.exists(self.attendance_file):
            try:
                df = pd.read_excel(self.attendance_file)
                for name in df['Name']:
                    if name in self.attendance_status:
                        self.attendance_status[name] = True
                self.update_attendance_status()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file Excel: {str(e)}")
                return

        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Đang điểm danh...")
        self.recognition_thread = threading.Thread(
            target=recognize_faces,
            args=(
                self.known_face_encodings,
                self.known_face_names,
                self.video_label,
                self,
                self.update_status,
                self.stop_event,
                self.attendance_status
            )
        )
        self.recognition_thread.daemon = True
        self.recognition_thread.start()

    def stop_recognition(self):
        self.stop_event.set()
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
            self.recognition_thread = None
        self.start_button.config(state=tk.NORMAL)
        self.add_face_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("Sẵn sàng")
        self.video_label.config(
            text="Chào mừng đến với hệ thống điểm danh!",
            image="",
            font=("Helvetica", 14),
            bg="#d3dce6",
            fg="#34495e",
            width=80,
            height=30
        )
        self.update_attendance_status()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()