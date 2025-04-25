import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
import os
from capture import collect_training_images
from train import train_face_recognition
from recognize import recognize_faces
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Điểm Danh Khuôn Mặt")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(True, True)

        self.known_face_encodings, self.known_face_names = train_face_recognition()
        self.stop_event = threading.Event()
        self.recognition_thread = None
        self.attendance_status = {name: False for name in self.known_face_names}

        self.main_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

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

        self.add_face_button = tk.Button(
            self.button_frame,
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
            self.button_frame,
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
            self.button_frame,
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
        training_dir = "training_faces"
        if not os.path.exists(training_dir):
            return "Số khuôn mặt đã thêm: 0"
        face_count = len([name for name in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, name))])
        return f"Số khuôn mặt đã thêm: {face_count}"

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
            self.known_face_encodings, self.known_face_names = train_face_recognition()
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

    def start_recognition(self):
        if self.recognition_thread and self.recognition_thread.is_alive():
            return

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.add_face_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
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