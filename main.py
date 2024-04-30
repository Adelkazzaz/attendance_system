import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from openpyxl import Workbook
from best_frame import choose_best_frame
from simpleFaceRec import SimpleFacerec
from database import *
import sqlite3

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        self.current_camera_index = 0
        self.attendance_started = False
        self.attendance_list = []
        
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("faces_database/")
        
        self.cap = cv2.VideoCapture(self.current_camera_index)

        self.label = tk.Label(root)
        self.label.pack(side=tk.TOP, pady=10)

        self.frame_buttons = tk.Frame(root)
        self.frame_buttons.pack(side=tk.BOTTOM, pady=10)

        self.quit_button = tk.Button(self.frame_buttons, text="Quit", command=self.quit, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.quit_button.pack(side=tk.LEFT, padx=10)

        self.switch_camera_button = tk.Button(self.frame_buttons, text="Switch Camera", command=self.switch_camera, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.switch_camera_button.pack(side=tk.LEFT, padx=10)

        self.add_student_button = tk.Button(self.frame_buttons, text="Add Student", command=self.add_student, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.add_student_button.pack(side=tk.LEFT, padx=10)

        self.start_attendance_button = tk.Button(self.frame_buttons, text="Start Attendance", command=self.start_attendance, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.start_attendance_button.pack(side=tk.LEFT, padx=10)

        self.stop_attendance_button = tk.Button(self.frame_buttons, text="Stop Attendance", command=self.stop_attendance, state=tk.DISABLED, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.stop_attendance_button.pack(side=tk.LEFT, padx=10)

        self.copy_names_button = tk.Button(self.frame_buttons, text="Save Names to Excel", command=self.save_names_to_excel, state=tk.DISABLED, bg="#C850C0", fg="white", padx=10, pady=5)  # Custom-styled button
        self.copy_names_button.pack(side=tk.LEFT, padx=10)

        self.attendees_label = tk.Label(root, text="Attendance: ", bg="lightgray", padx=10, pady=5)  # Custom-styled label
        self.attendees_label.pack(side=tk.BOTTOM, pady=10)

        # Frame for name and ID inputs
        self.student_info_frame = tk.Frame(root)
        self.student_info_frame.pack(side=tk.BOTTOM, pady=10)

        self.name_label = tk.Label(self.student_info_frame, text="Enter student's name: ", bg="lightgray")  # Custom-styled label
        self.name_label.pack(side=tk.LEFT)

        self.name_entry = tk.Entry(self.student_info_frame)
        self.name_entry.pack(side=tk.LEFT)

        self.id_label = tk.Label(self.student_info_frame, text="Enter student's ID: ", bg="lightgray")  # Custom-styled label
        self.id_label.pack(side=tk.LEFT)

        self.id_entry = tk.Entry(self.student_info_frame)
        self.id_entry.pack(side=tk.LEFT)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.attendance_started:
            face_locations, face_id = self.sfr.detect_known_faces(frame)
            face_names=get_Name_by_id(face_id)
            for name in face_names:
                if name not in self.attendance_list:
                    self.attendance_list.append(name)

            self.update_attendees_label()

        # Resize image to fit window
        height, width, _ = frame.shape
        if height > 500 or width > 700:
            frame = cv2.resize(frame, (700, 500))

        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.label.img = img
        self.label.config(image=img)
        
        self.root.after(10, self.update_frame)

    def switch_camera(self):
        self.cap.release()
        self.current_camera_index = (self.current_camera_index + 1) % 10
        self.cap = cv2.VideoCapture(self.current_camera_index)

    def start_attendance(self):
        self.attendance_started = True
        self.start_attendance_button.config(state=tk.DISABLED)
        self.stop_attendance_button.config(state=tk.NORMAL)
        self.copy_names_button.config(state=tk.DISABLED)
        self.attendance_list = []

    def stop_attendance(self):
        self.attendance_started = False
        self.start_attendance_button.config(state=tk.NORMAL)
        self.stop_attendance_button.config(state=tk.DISABLED)
        self.copy_names_button.config(state=tk.NORMAL)

    def save_names_to_excel(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["Names"])
        for name in self.attendance_list:
            
            sheet.append([name])
    
        excel_filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if excel_filename:
            workbook.save(excel_filename)
    def add_student(self):
        # Open a file dialog to choose the file
        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Video files", "*.mp4"), ("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return  # User canceled the selection

        # Get student name and ID from entry fields
        student_name = self.name_entry.get()
        student_id = self.id_entry.get()

        choose_best_frame(file_path, student_id)
        insert_student(student_name, student_id)

        # Clear the entry fields
        self.name_entry.delete(0, tk.END)
        self.id_entry.delete(0, tk.END)

    def update_attendees_label(self):
        names_str = ", ".join(self.attendance_list)
        self.attendees_label.config(text="Attendance: " + names_str)

    def quit(self):
        self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
