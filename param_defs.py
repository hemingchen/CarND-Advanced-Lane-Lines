import pickle

import cv2
import numpy as np

img_shape = (720, 1280, 3)

# Load calibration data
with open("calibration_data.p", mode='rb') as f:
    camera_calib = pickle.load(f)
mtx = camera_calib["mtx"]
dist = camera_calib["dist"]

# Define sliding window parameters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
# Number of windows from bottom to top
nwindows = 9
# Width of the windows +/- margin
margin = 100
# Minimum number of pixels found to recenter window
minpix = 50

# Define lane detection/marking parameters
step_size = 30
lane_color = [255, 255, 0]
lane_thickness = 10
highlight_lane_area_y_start = 0
highlight_lane_area_y_end = 720
highlight_color = [0, 255, 0]
vertices = np.array([[(0, img_shape[0]), (550, 470), (800, 470), (img_shape[1], img_shape[0])]], dtype=np.int32)
ksize = 3
xgrad_thresh = (20, 100)
ygrad_thresh = (70, 200)
mag_thresh = (70, 255)
dir_thresh = (0.1, 1.5) #(0, np.pi / 2)
h_thresh = None
l_thresh = None
s_thresh = (100, 255) # (170, 255)

# Define perspective transformation matrix
src = np.float32(
    [[100, 720],
     [550, 470],
     [700, 470],
     [1000, 720]])
dst = np.float32(
    [[200, 720],
     [200, 0],
     [1000, 0],
     [1000, 720]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

# Minimum valid curverad value
minfot = 1.0# 0.5
lane_width_pixel = 800

# Debug
debug = True
