import cv2
import numpy as np
from sklearn.cluster import KMeans

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Đọc frame đầu tiên
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

while True:
    # Đọc frame tiếp theo
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Tính toán sự khác biệt giữa hai frame
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Tìm các vùng có chuyển động
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Thêm đoạn mã KMeans để phân đoạn
    if contours:
        # Tạo một danh sách các điểm để phân đoạn
        points = np.array([cv2.boundingRect(c)[:2] for c in contours])
        kmeans = KMeans(n_clusters=min(3, len(points)))  # Số lượng cụm tối đa là 3
        kmeans.fit(points)
        labels = kmeans.labels_

        # Vẽ các hình chữ nhật cho từng cụm
        for i in range(max(labels) + 1):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                x_min, y_min = np.min(cluster_points, axis=0)
                x_max, y_max = np.max(cluster_points, axis=0)
                cv2.rectangle(frame2, (x_min, y_min), (x_max + 50, y_max + 50), (0, 0, 255), 2)  # Vẽ hình chữ nhật cho cụm

    # Hiển thị kết quả
    cv2.imshow('binary',gray2)
    cv2.imshow('Motion Detection', frame2)  # Hiển thị frame gốc với hình chữ nhật

    # Cập nhật frame trước
    gray1 = gray2

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
