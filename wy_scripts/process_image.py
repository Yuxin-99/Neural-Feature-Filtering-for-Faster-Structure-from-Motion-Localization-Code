import cv2
import numpy as np
import os
import sys

# python3 process_image.py ../CMU-models/images ../CMU-models/undistorted_imgs/
# python3 process_image.py ../CMU-models/query_images ../CMU-models/undistorted_query_imgs/

camera_intrinsics = np.array([[868.993378, 0, 525.942323],
                              [0, 866.063001, 420.042529],
                              [0,          0,          1]])

dist_coeff = np.array([[-0.399431, 0.188924, 0.000153, 0.000571]])


def main():
    img_path = sys.argv[1]
    img_names = os.listdir(img_path)
    output_path = sys.argv[2]

    for img_name in img_names:
        if "img" in img_name:
            image = cv2.imread(img_path + "/" + img_name)
            # set camera parameters
            height, width = image.shape[:2]

            # newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(camera_intrinsics, dist_coeff, (width, height), 1, (width, height))
            # x, y, w, h = validPixROI #(211, 991, 547, 755)
            # outputImage = outputImage[y-200:y+h+200, x-40:x+w+80] #fudge factor to minimize cropping

            undistorted_img = cv2.undistort(image, camera_intrinsics, dist_coeff)
            cv2.imwrite(output_path + img_name, undistorted_img)


if __name__ == "__main__":
    main()