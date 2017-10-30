import glob
import pickle

import cv2
import numpy as np

from helper_funcs import visualize_image_transformation

# Calibration points
nx = 9  # Number of inside corners in a row
ny = 6  # Number of inside corners in a column

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Test undistortion on one image
cal_img = cv2.imread('camera_cal/calibration1.jpg')
img_size = (cal_img.shape[1], cal_img.shape[0])

# Do calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

undistorted_cal_img = cv2.undistort(cal_img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg', undistorted_cal_img)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("calibration_data.p", "wb"))

# Visualize undistortion
visualize_image_transformation(
    before_img=cal_img, before_img_title='Original Chessboard Image',
    after_img=undistorted_cal_img, after_img_title='Undistorted Chessboard Image')