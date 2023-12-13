import os
import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe 手部模型
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 读取相机的投影矩阵
camera_matrices = []
for i in range(3):
    projection_matrix = np.loadtxt(f"./data/projection_matrix{i}.txt")
    camera_matrices.append(projection_matrix)

print(camera_matrices)

# 打开视频文件
video0 = cv2.VideoCapture('../data/camera_0.avi')
video1 = cv2.VideoCapture('../data/camera_1.avi')
video2 = cv2.VideoCapture('../data/camera_2.avi')

# 创建视频输出
output_video = cv2.VideoWriter('../data/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 480))

frames_success = []

results_list1 = []
with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands1:
    while video0.isOpened():
        ret0, frame0 = video0.read()
        if not ret0:
            break
        frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        results = hands1.process(frame_rgb)
        results_list1.append(results)

results_list2 = []
with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands2:
    while video1.isOpened():
        ret1, frame1 = video1.read()
        if not ret1:
            break
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        results = hands2.process(frame_rgb)
        results_list2.append(results)

video0.release()
video1.release()
video2.release()
cv2.destroyAllWindows()

video0 = cv2.VideoCapture('../data/camera_0.avi')
video1 = cv2.VideoCapture('../data/camera_1.avi')
video2 = cv2.VideoCapture('../data/camera_2.avi')

print(len(results_list1))

results_count = 0 
with mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands3:
    while video0.isOpened() and video1.isOpened() and video2.isOpened():
        ret0, frame0 = video0.read()
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret0 or not ret1 or not ret2:
            break

        results_list = []  # 每个视频帧中检测到的手的结果
        # 对视频0和视频1的每一帧进行手部关键点检测
        results_list.append(results_list1[results_count])
        results_list.append(results_list2[results_count])
        results_count += 1

        flag_continue = False
        for results in results_list:
            if not(results.multi_hand_landmarks):
                flag_continue = True
                break
            else:
                num_hands = len(results.multi_hand_landmarks)
                if num_hands != 2:
                    flag_continue = True
        if flag_continue:

            frames_success.append(False)
            continue
        else:
            frames_success.append(True)

        hand_left =  [[] for _ in range(21)]
        hand_right = [[] for _ in range(21)]

        for results, frame in zip(results_list, [frame0, frame1]):    
            if results.multi_hand_landmarks:
                # 遍历每只检测到的手
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    
                    # 获取手的标签（左手或右手）
                    label = handedness.classification[0].label

                    # 遍历所有关键点
                    for id, lm in enumerate(hand_landmarks.landmark):
                        # 提取归一化的 x 和 y 坐标
                        x, y = lm.x, lm.y

                        # 可以将这些坐标转换为相对于原始图像的像素坐标
                        h, w, _ = frame.shape  # 假设 frame 是原始帧
                        x_px, y_px = int(x * w), int(y * h)

                        # 如果是左手
                        if label == 'Left':
                            hand_left[id].append((x_px, y_px))
                        else:
                            hand_right[id].append((x_px, y_px))
        

                
        world_left_hand = []
        world_right_hand = []

        for i in range(21):
            points_on_images = hand_left[i]  # 每个相机视图中点的坐标   
            # 构建线性方程组的矩阵
            A = []
            for P, (x, y) in zip(camera_matrices[:-1], points_on_images):

                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])

            A = np.array(A)

            # 使用SVD求解线性方程组
            U, S, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[-1]  # 将齐次坐标转换为非齐次坐标

            world_point = X[:3]  # 世界坐标系中的点
            world_left_hand.append(world_point)
        for i in range(21):
            points_on_images = hand_right[i]
            # 构建线性方程组的矩阵
            A = []
            for P, (x, y) in zip(camera_matrices[:-1], points_on_images):
                A.append(x * P[2, :] - P[0, :])
                A.append(y * P[2, :] - P[1, :])
            
            A = np.array(A)

            # 使用SVD求解线性方程组
            U, S, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X / X[-1]

            world_point = X[:3]
            world_right_hand.append(world_point)

        # 使用相机2的投影矩阵将3D坐标投射回2D
        for world_point in world_left_hand:
            # 将世界坐标系中的点转换为齐次坐标
            world_point_homogeneous = np.append(world_point, 1)
            
            # 使用相机2的投影矩阵将3D点投影回2D
            projected_point = np.dot(camera_matrices[2], world_point_homogeneous)
            
            # 将齐次坐标转换为非齐次坐标
            projected_point = projected_point[:2] / projected_point[2]
            
            # 在视频2的帧上绘制投影点
            cv2.circle(frame2, (int(projected_point[0]), int(projected_point[1])), 3, (0, 255, 0), -1)
        for world_point in world_right_hand:
            # 将世界坐标系中的点转换为齐次坐标
            world_point_homogeneous = np.append(world_point, 1)
            
            # 使用相机2的投影矩阵将3D点投影回2D
            projected_point = np.dot(camera_matrices[2], world_point_homogeneous)
            
            # 将齐次坐标转换为非齐次坐标
            projected_point = projected_point[:2] / projected_point[2]
            
            # 在视频2的帧上绘制投影点
            cv2.circle(frame2, (int(projected_point[0]), int(projected_point[1])), 3, (255, 0, 0), -1)
    

        # 将处理后的帧写入输出视频
        output_video.write(frame2)

# 释放资源
video0.release()
video1.release()
video2.release()
output_video.release()
cv2.destroyAllWindows()
