import cv2
import threading
import tkinter as tk
from tkinter import messagebox, ttk

output_path = "digit_camera_output.mp4"
frame_width = 320
frame_height = 240
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')

recording = False
selected_camera_index = 1

def start_recording():
    global recording, selected_camera_index
    if recording:
        return
    recording = True
    threading.Thread(target=record_video, args=(selected_camera_index,)).start()

def record_video(cam_index):
    global recording

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)



    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print("Recording started. Press 'q' in the video window to stop")

    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Failed.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('DIGIT Camera View', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved to {output_path}")
    recording = False

def on_camera_select(event):
    global selected_camera_index
    selected_camera_index = int(camera_dropdown.get())

def get_available_cameras():
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available

# GUI setup
root = tk.Tk()
root.title("DIGIT Sensor Recorder")

tk.Label(root, text="Select Camera Index:", font=("Helvetica", 12)).pack(pady=5)
camera_dropdown = ttk.Combobox(root, values=get_available_cameras(), state="readonly")
camera_dropdown.current(0)
camera_dropdown.bind("<<ComboboxSelected>>", on_camera_select)
camera_dropdown.pack(pady=5)

start_button = tk.Button(root, text="Start Recording", command=start_recording, font=("Helvetica", 16))
start_button.pack(pady=20)

root.mainloop()
