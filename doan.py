import cv2
import numpy as np
import time

# Khởi tạo camera
# cap = cv2.VideoCapture("6110351164988.mp4")

cap = cv2.VideoCapture(0)

# Thiết lập optical flow Farneback
farneback_params = dict(pyr_scale=0.5, levels=5, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# Đọc khung hình đầu tiên
ret, old_frame = cap.read()
if not ret:
    print("Không thể khởi động camera.")
    exit()
# Chuyển khung hình sang grayscale
scale_factor = 0.5
old_frame = cv2.resize(old_frame, (0, 0), fx=scale_factor, fy=scale_factor)  # Scale the frame

h,w,_=old_frame.shape


center_point_x = w // 2  # Tính toán tọa độ x trung tâm
center_point_y = h // 2  # Tính toán tọa độ y trung tâm


hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Thông số lưới
step = 15  # Khoảng cách giữa các điểm trong lưới
h, w = old_gray.shape

# Hàm khởi tạo lưới
def initialize_grid(step, h, w):
    x, y = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    return np.stack((x.ravel(), y.ravel()), axis=-1)

# Tạo lưới cố định
grid_points = initialize_grid(step, h, w)

# Tạo mặt nạ để vẽ đường quỹ đạo
mask = np.zeros_like(old_frame)

# Thời điểm bắt đầu
last_reset_time = time.time()
reset_interval = 5  # Thời gian xóa đường quỹ đạo (giây)

while cap.isOpened():
    ret, frame = cap.read()


    if not ret:
        cap = cv2.VideoCapture("6110351164988.mp4")
        ret, frame = cap.read()
        center_point_x = w // 2  # Tính toán tọa độ x trung tâm
        center_point_y = h // 2  # Tính toán tọa độ y trung tâm


        # break
    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)  # Scale the frame

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)


    # Tính toán optical flow Farneback
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **farneback_params)

    # Vẽ đường chuyển động (giữ lưới cố định)
    total_new_x=0
    total_new_y=0
    total_pt_x=0
    total_pt_y=0
    total_fx=0
    total_fy=0
    count=1
    
    for pt in grid_points:
        fx, fy = flow[pt[1], pt[0]]  # Vector quang học tại mỗi điểm
        if (abs(fy)>0.5) or (abs(fx)>0.5):
            new_x = int(pt[0] + fx)
            new_y = int(pt[1] + fy)
            # Đảm bảo vector trong giới hạn khung hình
            new_x = np.clip(new_x, 0, w - 1)
            new_y = np.clip(new_y, 0, h - 1)

            # total_new_x+=new_x
            # total_new_y+=new_y
            # total_pt_x+=pt[0]
            # total_pt_y+=pt[1]
            total_fx+=fx
            total_fy+=fy
            count+=1

            # Vẽ quỹ đạo chuyển động
            # cv2.line(mask, (pt[0], pt[1]), (new_x, new_y), (0, 255, 0), 1)
            frame = cv2.arrowedLine(frame, (pt[0], pt[1]), (new_x, new_y), (0, 255, 0), 1, tipLength=0.5)

            # cv2.circle(frame, (new_x, new_y), 2, (0, 0, 255), -1)

    center_point_x=int(center_point_x+(total_fx/count))
    center_point_y=int(center_point_y+(total_fy/count))
    
    cv2.circle(frame, (center_point_x, center_point_y), 7, (0, 0, 255), -1)

    # Hiển thị khung hình với lưới và quỹ đạo
    img = cv2.add(frame, mask)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Hiển thị
    cv2.imshow('frame', rgb)
    # cv2.imshow('',old_frame)
    cv2.imshow('Optical Flow Grid', img)

    # In ra tọa độ di chuyển

    # Xóa các đường quỹ đạo cũ sau 4-5 giây
    current_time = time.time()
    if current_time - last_reset_time > reset_interval:
        mask = np.zeros_like(old_frame)  # Đặt lại mặt nạ
        last_reset_time = current_time  # Cập nhật thời gian đặt lại
     #
    # Cập nhật khung hình
    old_gray = frame_gray.copy()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
