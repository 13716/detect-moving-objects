import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.cluster import KMeans

# Function to perform K-means clustering on the frame
def process_frame(frame, k):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data = np.array(img).reshape((-1, 3))

    # K-means clustering
    kmeans = KMeans(n_clusters=k, init='random', n_init=1, max_iter=100, random_state=0)
    start_time = time.time()
    kmeans.fit(data)
    end_time = time.time()
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.astype(np.uint8)

    segmented_image = centroids[labels].reshape(img.shape)  # Reconstruct the image
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)  # Data type conversion
    
    # Create binary image based on the cluster centroids
    if np.mean(centroids[0]) > np.mean(centroids[1]):
        binary_img = np.where(labels.reshape(img.shape[:2]) == 0, 255, 0).astype(np.uint8)
    else:
        binary_img = np.where(labels.reshape(img.shape[:2]) == 1, 255, 0).astype(np.uint8)

    # Erosion for noise reduction
    kernel = np.ones((5, 5), np.uint8)
    new = cv2.erode(segmented_image, kernel, iterations=1)
    img2 = np.zeros_like(img)
    img2[(new >= 100) & (new < 255)] = 255
    
    return binary_img, segmented_image, img2, end_time - start_time

# Function to detect motion between frames
def detect_motion(prev_frame, current_frame):
    # Convert frames to grayscale
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the frames
    diff = cv2.absdiff(gray_prev, gray_current)

    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the motion areas
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return contours for further processing
    return contours

# Function to update the GUI with the camera feed and processed images
def update_frame():
    global cap, prev_frame
    ret, frame = cap.read()
    if ret:
        # Detect motion
        if prev_frame is not None:
            contours = detect_motion(prev_frame, frame)
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Chỉ giữ lại các vùng lớn hơn 500 pixel
                    (x, y, w, h) = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

            # Vẽ hình chữ nhật lớn bao quanh tất cả các đối tượng
            if x_max > x_min and y_max > y_min:  # Kiểm tra nếu có ít nhất một đối tượng
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Vẽ hình chữ nhật lớn
                print(f"Vị trí của vật thể di chuyển: ({x_min}, {y_min}) đến ({x_max}, {y_max})")  # In ra vị trí

        prev_frame = frame  # Update previous frame

        try:
            k = int(entry_k.get())  # Dynamically read k from the entry box
        except ValueError:
            k = 2  # Default to 2 if invalid input

        # Avoid extremely low/high k values
        if k < 2:
            k = 2
        elif k > 20:
            k = 20

        binary_img, segmented_image, img2, exec_time = process_frame(frame, k)

        # Convert processed images to ImageTk format for displaying in tkinter
        binary_img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        segmented_img_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        new_img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        binary_pil = Image.fromarray(binary_img_rgb)
        segmented_pil = Image.fromarray(segmented_img_rgb)
        new_pil = Image.fromarray(new_img_rgb)

        binary_tk = ImageTk.PhotoImage(binary_pil)
        segmented_tk = ImageTk.PhotoImage(segmented_pil)
        new_tk = ImageTk.PhotoImage(new_pil)

        # Update the canvas images
        canvas_binary.create_image(0, 0, anchor=tk.NW, image=binary_tk)
        canvas_binary.image = binary_tk  # To prevent garbage collection

        canvas_segmented.create_image(0, 0, anchor=tk.NW, image=segmented_tk)
        canvas_segmented.image = segmented_tk  # To prevent garbage collection

        canvas_new.create_image(0, 0, anchor=tk.NW, image=new_tk)
        canvas_new.image = new_tk  # To prevent garbage collection

        # Call update_frame again after 10ms
        root.after(10, update_frame)

# Start the camera automatically
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Tắt tự động điều chỉnh độ phơi sáng
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
    update_frame()

# Stop the camera and close the application when the window is closed
def on_closing():
    global cap
    if cap is not None:
        cap.release()
    root.destroy()

# Main GUI setup
root = tk.Tk()
root.title("Camera K-Means Clustering")

# Input for k
tk.Label(root, text="Number of clusters (k):").pack(pady=5)
entry_k = tk.Entry(root)
entry_k.pack(pady=5)
entry_k.insert(0, '2')  # Set default value for k

# Create canvas to display the images side by side
canvas_frame = tk.Frame(root)
canvas_frame.pack()

canvas_binary = tk.Canvas(canvas_frame, width=480, height=480)
canvas_binary.pack(side=tk.LEFT, padx=10, pady=10)

canvas_segmented = tk.Canvas(canvas_frame, width=480, height=480)
canvas_segmented.pack(side=tk.LEFT, padx=10, pady=10)

canvas_new = tk.Canvas(canvas_frame, width=480, height=480)
canvas_new.pack(side=tk.LEFT, padx=10, pady=10)

# Global variables
cap = None
prev_frame = None

# Start camera automatically on startup
start_camera()

# Handle the closing of the window
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the main GUI loop
root.mainloop()
