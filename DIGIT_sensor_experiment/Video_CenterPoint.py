import cv2
import os 
# Calibration
img_width_px = 320  
img_height_px = 240
width_mm = 20.0
height_mm = 15.0
scale_x = width_mm / img_width_px     # 0.03125 mm/px
scale_y = height_mm / img_height_px   # 0.03125 mm/px

origin_px = (0, 120)  # (x, y)

# Load image
project_dir = os.path.dirname(os.path.abspath(__file__))
indenter_name= 'edge6'
image_name = f"output\{indenter_name}\{indenter_name}_90.jpg"
image_path = os.path.join(project_dir, image_name)

img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Draw '+' marker at the origin
cv2.drawMarker(img, origin_px, color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
               markerSize=20, thickness=2)

# Mouse click callback: Convert to mm with flipped Y-axis
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        dx = x - origin_px[0]
        dy = origin_px[1] - y  # Flip y-axis: up is positive
        y_mm = dx * scale_x
        x_mm = -dy * scale_y
        print(f"Pixel: ({x}, {y}) → Physical: ({x_mm:.3f} mm, {y_mm:.3f} mm)")

# Display image with probe interaction
cv2.namedWindow("Probe DIGIT image")
cv2.setMouseCallback("Probe DIGIT image", on_mouse_click)
cv2.imshow("Probe DIGIT image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
