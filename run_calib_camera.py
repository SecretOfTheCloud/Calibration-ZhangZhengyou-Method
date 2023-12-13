import cv2
import os

def capture_images(camera_list, num_images):

    caps = []
    for camera in camera_list:
        camera_id = camera[0]
        camera_name = camera[1]
        cap = cv2.VideoCapture(camera_id)
        caps.append(cap)
        folder_name = f"./pic/testCamera{camera_id}"
        os.makedirs(folder_name, exist_ok=True)

    while True:      
        camera_id = 0 
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera{camera_id}", frame)
                camera_id += 1
            else:
                print(f"Failed to capture image from camera {camera_name}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for i in range(num_images):
        for j, camera in enumerate(camera_list):
            camera_id = camera[0]
            camera_name = camera[1]
            folder_name = f"./pic/testCamera{camera_id}"

            cap = caps[j]        
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{folder_name}/{camera_name}_{i+1}.png", frame)
                cv2.imshow(f"Camera{camera_id}", frame)
            else:
                print(f"Failed to capture image from camera {camera_name}")
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    for cap in caps:
        cap.release()

camera_list = [(0, "Camera1"), (1, "Camera2"), (2, "Camera3")]
num_images = 10

capture_images(camera_list, num_images)
