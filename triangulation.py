import numpy as np
import cv2

# 假设 camera_matrices 是一个包含各个相机投影矩阵的列表
# points_on_images 是一个列表，包含该点在每个相机视图中的坐标 [(x1, y1), (x2, y2), ...]
camera_matrices = []  # 每个相机的投影研阵
img_dirs = ["./pic/testCamera0", "./pic/testCamera1", "./pic/testCamera2"]
for i, img_dir in enumerate(img_dirs):
    projection_matrix = np.loadtxt("./data/projection_matrix"+str(i)+".txt")
    camera_matrices.append(projection_matrix)

points_on_images = [...]  # 每个相机视图中点的坐标

# 构建线性方程组的矩阵
A = []
for P, (x, y) in zip(camera_matrices, points_on_images):
    A.append(x * P[2, :] - P[0, :])
    A.append(y * P[2, :] - P[1, :])

A = np.array(A)

# 使用SVD求解线性方程组
U, S, Vt = np.linalg.svd(A)
X = Vt[-1]
X = X / X[-1]  # 将齐次坐标转换为非齐次坐标

world_point = X[:3]  # 世界坐标系中的点
