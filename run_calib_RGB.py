# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/RGB_camera_calib_img/", Without Distortion. 

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os
import cv2
import numpy as np

from calibrate_helper import Calibrator



def main():
    # img_dir = "./pic/RGB_camera_calib_img"
    img_dirs = ["./pic/testCamera0", "./pic/testCamera1", "./pic/testCamera2"]

    for count, img_dir in enumerate(img_dirs):

        shape_inner_corner = (8, 5)
        size_grid = 0.027
        # create calibrator
        calibrator = Calibrator(img_dir, shape_inner_corner, size_grid)
        # calibrate the camera
        mat_intri, coff_dis, v_rot, v_trans = calibrator.calibrate_camera()
        rotation_matrix, _ = cv2.Rodrigues(v_rot)

        # Calculate the extrinsic matrix
        extrinsic_matrix = cv2.hconcat([rotation_matrix, v_trans])
        print("extrinsic_matrix:\n", extrinsic_matrix)

        # Calculate the projection matrix
        projection_matrix = mat_intri @ extrinsic_matrix
        print("projection_matrix:\n", projection_matrix)
        print(projection_matrix.shape)


        #有一个世界系坐标（1,1,1），求其在图像上的坐标，并绘制在img_dir的第一张图上验证
        world_coordinate = [0, 0, 0]
        world_coordinate = np.array([world_coordinate])
        world_coordinate = cv2.convertPointsToHomogeneous(world_coordinate)
        print("world_coordinate:\n", world_coordinate)
        image_coordinate = projection_matrix @ world_coordinate[0].T
        print("image_coordinate:\n", image_coordinate)

        image_coordinate=np.array(image_coordinate.T)
        image_coordinate = cv2.convertPointsFromHomogeneous(image_coordinate)
        print("image_coordinate:\n", image_coordinate)

        img = cv2.imread(os.path.join(img_dir, "Camera"+str(count+1)+"_1.png"))
        img = cv2.circle(img, (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1])), 2, (0, 0, 255), -1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #把投射矩阵保存到文件中
        np.savetxt("./data/projection_matrix"+str(count)+".txt", projection_matrix)

        #从"./data/projection_matrix.txt"该文件读取投射矩阵
        pro_matrix = np.loadtxt("./data/projection_matrix"+str(count)+".txt")
        print("test: ",pro_matrix)
    

if __name__ == '__main__':
    main()
