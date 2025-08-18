import cv2
import os
import numpy as np

# Paths 
indenter_name= 'edge7'
video_path = f"DIGIT_sensor_experiment\EditedVideos\{indenter_name}.mp4"
output_dir = f"DIGIT_sensor_experiment\ImageFrames\{indenter_name}"

os.makedirs(output_dir, exist_ok=True)
# Load the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Choose N frame indices evenly spaced
Number_of_Images = 100
sample_indices = np.linspace(0, total_frames - 30, Number_of_Images, dtype=int)

# Loop and save selected frames
saved = 0
for idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    if idx in sample_indices:
        filename = os.path.join(output_dir, f"{indenter_name}_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1  

cap.release()
print(f"Saved {saved} frames.")
