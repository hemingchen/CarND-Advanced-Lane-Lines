import matplotlib.image as mpimg

from helper_funcs import *
from param_defs import *

# raw_img = mpimg.imread("test_images/project_video_41_42/7.jpg")
# raw_img = mpimg.imread("test_images/challenge_video_0_1/2.jpg")
raw_img = mpimg.imread("test_images/test1.jpg")

###############################################################################
# 1) Test undistortion
###############################################################################
undistorted_raw_img = cv2.undistort(raw_img, mtx, dist, None, mtx)

visualize_image_transformation(
    before_img=raw_img, before_img_title='Original Image',
    after_img=undistorted_raw_img, after_img_title='Undistorted Image')

###############################################################################
# 2) Test gradient thresholding functions
###############################################################################
gradx = abs_sobel_thresh(undistorted_raw_img, orient='x', ksize=ksize, thresh=xgrad_thresh)
grady = abs_sobel_thresh(undistorted_raw_img, orient='y', ksize=ksize, thresh=ygrad_thresh)

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=gradx, after_img_title='Absolute Gradient Thresholding in x Direction')
visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=grady, after_img_title='Absolute Gradient Thresholding in y Direction')

###############################################################################
# 3) Test magnitude of gradient thresholding function
###############################################################################
mag_binary = mag_threshold(undistorted_raw_img, ksize=ksize, thresh=mag_thresh)

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=mag_binary, after_img_title='Magnitude of Gradient Thresholding')

###############################################################################
# 4) Test directional gradient thresholding function
###############################################################################
dir_binary = dir_threshold(undistorted_raw_img, ksize=ksize, thresh=dir_thresh)

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=dir_binary, after_img_title='Direction of Gradient Thresholding')

###############################################################################
# 5) Test color channel thresholding function
###############################################################################
color_binary = color_channel_threshold(undistorted_raw_img, color_channel='s', thresh=s_thresh)

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=color_binary, after_img_title='S Channel Thresholding')

###############################################################################
# 6) Test combination of all binary images
###############################################################################
combined_binary = np.zeros_like(dir_binary, dtype=np.uint8)
combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | color_binary == 1] = 1

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=combined_binary, after_img_title='Combination of Gradient, Mag, Dir Thresholding')

###############################################################################
# 7) Test selected combination of thresholds
###############################################################################
combined_binary = apply_multiple_thresholds(
    img=undistorted_raw_img,
    ksize=3,
    xgrad_thresh=(20, 100),
    ygrad_thresh=(70, 200),
    mag_thresh=(70, 255),
    dir_thresh=(0, np.pi / 2),
    h_thresh=None,
    l_thresh=None,
    s_thresh=(170, 255))

visualize_image_transformation(
    before_img=undistorted_raw_img, before_img_title='Original Image',
    after_img=combined_binary, after_img_title='Combination of Selected Thresholds')

###############################################################################
# 8) Test masking
###############################################################################
masked_img = region_of_interest(combined_binary, vertices)

visualize_image_transformation(
    before_img=combined_binary, before_img_title='Before Applying Mask',
    after_img=masked_img, after_img_title='After Applying Mask')

###############################################################################
# 9) Test perspective transformation
###############################################################################
warped_img = cv2.warpPerspective(masked_img, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)

visualize_image_transformation(
    before_img=masked_img, before_img_title='Before Perspective Transformation',
    after_img=warped_img, after_img_title='After Perspective Transformation')

###############################################################################
# 10) Test lane line dection with sliding windows
###############################################################################
prior_left_fit, prior_right_fit = None, None

# Get starting points on the bottom of the image
leftx_base, rightx_base = get_starting_points_of_lane_lines(img=warped_img)

# Get lane lines with sliding windows
left_fit, right_fit, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img = \
    get_lane_lines_with_sliding_window(
        img=warped_img,
        leftx_base=leftx_base,
        rightx_base=rightx_base,
        prior_left_fit=prior_left_fit,
        prior_right_fit=prior_right_fit,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix,
        nwindows=nwindows,
        margin=margin,
        minpix=minpix,
        minfot=minfot,
        lane_width_pixel=lane_width_pixel,
        step_size=step_size,
        lane_line_color=lane_color,
        lane_line_thickness=lane_thickness)

plt.figure(figsize=(10, 5))
plt.imshow(draw_on_img)
title = "Lane Lines Detected by Sliding Windows"
plt.title(title, fontsize=20)
plt.savefig("./examples/" + title.replace(" ", "_") + ".jpg")

###############################################################################
# 11) Test lane line detection using prior frame as reference
###############################################################################
# Prior frame (although here we are using the same image for both frames)
# Get starting points on the bottom of the image
leftx_base, rightx_base = get_starting_points_of_lane_lines(
    img=warped_img)

# Get lane lines with sliding winows
prior_left_fit, prior_right_fit, _, _, _, _ = get_lane_lines_with_sliding_window(
    img=warped_img,
    leftx_base=leftx_base,
    rightx_base=rightx_base,
    prior_left_fit=prior_left_fit,
    prior_right_fit=prior_right_fit,
    xm_per_pix=xm_per_pix,
    ym_per_pix=ym_per_pix,
    nwindows=nwindows,
    margin=margin,
    minpix=minpix,
    minfot=minfot,
    lane_width_pixel=lane_width_pixel,
    step_size=step_size,
    lane_line_color=lane_color,
    lane_line_thickness=lane_thickness)

# Get lane lines using prior window information
left_fit, right_fit, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img = get_lane_lines_based_on_prior_frame_lines(
    img=warped_img,
    prior_left_fit=prior_left_fit,
    prior_right_fit=prior_right_fit,
    xm_per_pix=xm_per_pix,
    ym_per_pix=ym_per_pix,
    margin=margin,
    minfot=minfot,
    lane_width_pixel=lane_width_pixel,
    step_size=step_size,
    lane_line_color=lane_color,
    lane_line_thickness=lane_thickness)

plt.figure(figsize=(10, 5))
plt.imshow(draw_on_img)
plt.title("Lane Detection Based on Prior Window information", fontsize=20)

# Print out curvature
print("left line curvature: {:.2f} m".format(left_curverad))
print("right line curvature: {:.2f} m".format(right_curverad))

# Print out vehicle offset to the center of lane
print("vehicle is {:.2f} m {} of the center".format(veh_pos_wrt_ln_ctr, 'R' if veh_pos_wrt_ln_ctr < 0 else 'L'))

###############################################################################
# 12) Draw lane lines
###############################################################################
blank_canvas = np.zeros_like(raw_img)

lane_img = draw_lanes_on_img(
    img=blank_canvas,
    left_fit=left_fit,
    right_fit=right_fit,
    step_size=step_size,
    color=lane_color,
    thickness=lane_thickness,
    dashed=False)

plt.figure(figsize=(10, 5))
plt.imshow(lane_img)
plt.title("Lane Lines from Top View", fontsize=20)

###############################################################################
# 13) Highlight lane area
###############################################################################
hightlight_area_mask = highlight_lane_area(
    img=blank_canvas,
    left_fit=left_fit,
    right_fit=right_fit,
    y_start=highlight_lane_area_y_start,
    y_end=highlight_lane_area_y_end,
    highlight_color=highlight_color)

highlighted_lane_img = cv2.add(lane_img, hightlight_area_mask)

plt.figure(figsize=(10, 5))
plt.imshow(highlighted_lane_img)
title = "Hightlighted Lane Area from Top View"
plt.title(title, fontsize=20)
plt.savefig("./examples/" + title.replace(" ", "_") + ".jpg")

###############################################################################
# 14) Highlight lane area changed back to driver perspective
###############################################################################
lane_lines_driver_perspective = cv2.warpPerspective(
    highlighted_lane_img,
    Minv,
    (raw_img.shape[1], raw_img.shape[0]),
    flags=cv2.INTER_LINEAR)

plt.figure(figsize=(10, 5))
plt.imshow(lane_lines_driver_perspective)
plt.title("Lane Lines from Driver Perspective", fontsize=20)

###############################################################################
# 15) Highlight lane area on original (undistorted) image
###############################################################################
undistorted_raw_img_with_lanes = cv2.add(lane_lines_driver_perspective, undistorted_raw_img)

plt.figure(figsize=(10, 5))
plt.imshow(undistorted_raw_img_with_lanes)
title = "Detected Lane on Original Image"
plt.title(title, fontsize=20)
plt.savefig("./examples/" + title.replace(" ", "_") + ".jpg")

###############################################################################
# 15) Add lane information on image
###############################################################################
undistorted_raw_img_with_lanes_info = add_lane_info_to_image(
    img=undistorted_raw_img_with_lanes,
    left_curvature=left_curverad,
    right_curvature=right_curverad,
    veh_pos_wrt_ln_ctr=veh_pos_wrt_ln_ctr)

plt.figure(figsize=(10, 5))
plt.imshow(undistorted_raw_img_with_lanes_info)
title = "Lane Lines with Additional Information"
plt.title(title, fontsize=20)
plt.savefig("./examples/" + title.replace(" ", "_") + ".jpg")

###############################################################################
# 16) Test debug image output
###############################################################################
debug_img_1 = np.dstack((masked_img, masked_img, masked_img)) * 255
debug_img_2 = draw_on_img
final_img = add_debug_imgs(undistorted_raw_img_with_lanes_info, debug_img_1, debug_img_2)

plt.figure(figsize=(10, 5))
plt.imshow(final_img)
plt.title("Debug Image", fontsize=20)

###############################################################################
# End
###############################################################################
plt.show()
