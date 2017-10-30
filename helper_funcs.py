import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Visualization helper function
def visualize_image_transformation(before_img, before_img_title, after_img, after_img_title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(before_img)
    ax1.set_title(before_img_title, fontsize=20)
    ax2.imshow(after_img)
    ax2.set_title(after_img_title, fontsize=20)
    fig_path = "./examples/" + before_img_title.replace(" ", "_") + "_vs_" + after_img_title.replace(" ", "_") + ".jpg"
    plt.savefig(fig_path)


def abs_sobel_thresh(img, orient='x', ksize=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        print("Error: orient must be either x or y.")

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # So there are 1s where #s are within our thresholds and 0s otherwise.
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return grad_binary


def mag_threshold(img, ksize=9, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # 6) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # 7) Return this mask as your binary_output image
    return mag_binary


def dir_threshold(img, ksize=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return dir_binary


def color_channel_threshold(img, color_channel='s', thresh=(170, 255)):
    # Apply the following steps to img
    # 1) Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # 2) Separate by color channel = 'h', 'l' or 's'
    if color_channel == 'h':
        channel = hls[:, :, 0]
    elif color_channel == 'l':
        channel = hls[:, :, 1]
    elif color_channel == 's':
        channel = hls[:, :, 2]
    else:
        print("Error: orient must be either h, l or s.")

    # Cont'd: Threshold colour channel
    channel_binary = np.zeros_like(channel)
    channel_binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return channel_binary


def select_color_threshold(img, range, color_conversion=None):
    working_copy = np.copy(img)

    # Convert color space if needed
    if color_conversion is not None:
        working_copy = cv2.cvtColor(working_copy, color_conversion)
    # Select color per threshold
    select_color_binary = cv2.inRange(working_copy, np.array(range[0]), np.array(range[1]))

    return select_color_binary


def apply_multiple_thresholds(
        img,
        ksize=3,
        xgrad_thresh=(20, 100),
        ygrad_thresh=(70, 200),
        mag_thresh=(70, 255),
        dir_thresh=(0, np.pi / 2),
        h_thresh=None,
        l_thresh=None,
        s_thresh=(170, 255),
        yellow_range=([20, 60, 60], [38, 174, 250]),
        white_range=([202, 202, 202], [255, 255, 255]),
        black_range=([0, 0, 0], [30, 30, 30])):
    # 1) Add Sobel thresholds in x, y direction
    gradx = abs_sobel_thresh(img, orient='x', ksize=ksize, thresh=xgrad_thresh)
    grady = abs_sobel_thresh(img, orient='y', ksize=ksize, thresh=ygrad_thresh)

    # 2) Add directional thresholds
    dir_binary = dir_threshold(img, ksize=ksize, thresh=dir_thresh)

    # 3) Add magnitude of gradient thresholding
    mag_binary = mag_threshold(img, ksize=ksize, thresh=mag_thresh)

    # 4) Add s channel thresholds
    color_binary = color_channel_threshold(img, color_channel='s', thresh=s_thresh)

    # 5) Add yellow color selection thresholds
    select_yellow_binary = select_color_threshold(img, range=yellow_range, color_conversion=cv2.COLOR_RGB2HSV)

    # 6) Add white color selection thresholds
    select_white_binary = select_color_threshold(img, range=white_range, color_conversion=None)

    # 7) Add black color selection thresholds
    select_black_binary = select_color_threshold(img, range=black_range, color_conversion=None)

    # 8) Combine all binary thresholds
    combined_binary = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # combined_binary[
    #     (color_binary == 1) |
    #     (gradx == 1) |
    #     (select_yellow_binary == 1) |
    #     (select_white_binary == 1)] = 1
    # combined_binary[
    #     (gradx == 1) |
    #     ((select_yellow_binary == 1) & (color_binary == 1)) |
    #     (select_white_binary == 1)] = 1
    combined_binary[
        (gradx == 1) |
        (select_yellow_binary == 1) |
        ((color_binary == 1) & (select_black_binary == 0)) |
        (select_white_binary == 1)] = 1

    return combined_binary


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Define a blank mask to start with
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on the image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def get_starting_points_of_lane_lines(img):
    # Assuming input is a binary warped image
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base


def get_lane_lines_with_sliding_window(
        img, leftx_base, rightx_base, prior_left_fit, prior_right_fit, prior_left_fit_cr, prior_right_fit_cr,
        xm_per_pix, ym_per_pix, nwindows=9, margin=100, minpix=50, minfot=0.5, lane_width_pixel=800,
        step_size=30, lane_line_color=[255, 255, 0], lane_line_thickness=10):
    # Assuming input is a binary warped image
    # Create an output image to draw on and  visualize the result
    draw_on_img = np.dstack((img, img, img)) * 255

    # y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(draw_on_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(draw_on_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calibrate lane curve
    left_fit, right_fit, left_fit_cr, right_fit_cr = get_calibrated_lane_curves(
        prior_left_fit=prior_left_fit,
        prior_right_fit=prior_right_fit,
        prior_left_fit_cr=prior_left_fit_cr,
        prior_right_fit_cr=prior_right_fit_cr,
        left_fit=left_fit,
        right_fit=right_fit,
        left_fit_cr=left_fit_cr,
        right_fit_cr=right_fit_cr,
        xm_per_pix=xm_per_pix,
        minfot=minfot,
        lane_width_pixel=lane_width_pixel)

    # Add colors to draw_on_img
    draw_on_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    draw_on_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Add lane lines to draw_on_img
    lane_lines = draw_lanes_on_img(
        img=draw_on_img,
        left_fit=left_fit,
        right_fit=right_fit,
        step_size=step_size,
        color=lane_line_color,
        thickness=lane_line_thickness,
        dashed=False)
    draw_on_img = cv2.add(lane_lines, draw_on_img)

    # Get lane curvature
    left_curverad, right_curverad = get_radius_of_curvature(
        ploty=ploty,
        left_fit=left_fit_cr,
        right_fit=right_fit_cr,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix)

    # Get vehicle position wrt lane center
    veh_pos_wrt_ln_ctr = get_vehicle_position_wrt_lane_center(
        imshape=img.shape,
        left_fit=left_fit,
        right_fit=right_fit,
        left_curverad=left_curverad,
        right_curverad=right_curverad,
        xm_per_pix=xm_per_pix)

    return left_fit, right_fit, left_fit_cr, right_fit_cr, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img


def get_lane_lines_based_on_prior_frame_lines(
        img, prior_left_fit, prior_right_fit, prior_left_fit_cr, prior_right_fit_cr,
        xm_per_pix, ym_per_pix, margin=100, minfot=0.5, lane_width_pixel=800,
        step_size=30, lane_line_color=[255, 255, 0], lane_line_thickness=10):
    # Assume input is a binary warped image
    # Create an output image to draw on and  visualize the result
    draw_on_img = np.dstack((img, img, img)) * 255
    # Image to plot the search region on
    window_img = np.zeros_like(draw_on_img)

    # y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # Use lines detected in prior frame as reference for search in current frame
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = (
        (nonzerox > (prior_left_fit[0] * (nonzeroy ** 2) + prior_left_fit[1] * nonzeroy + prior_left_fit[2] - margin)) &
        (nonzerox < (prior_left_fit[0] * (nonzeroy ** 2) + prior_left_fit[1] * nonzeroy + prior_left_fit[2] + margin)))

    right_lane_inds = (
        (nonzerox > (
            prior_right_fit[0] * (nonzeroy ** 2) + prior_right_fit[1] * nonzeroy + prior_right_fit[2] - margin)) &
        (nonzerox < (
            prior_right_fit[0] * (nonzeroy ** 2) + prior_right_fit[1] * nonzeroy + prior_right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Sanity check
    left_fit = None
    right_fit = None

    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calibrate lane curve
        left_fit, right_fit, left_fit_cr, right_fit_cr = get_calibrated_lane_curves(
            prior_left_fit=prior_left_fit,
            prior_right_fit=prior_right_fit,
            prior_left_fit_cr=prior_left_fit_cr,
            prior_right_fit_cr=prior_right_fit_cr,
            left_fit=left_fit,
            right_fit=right_fit,
            left_fit_cr=left_fit_cr,
            right_fit_cr=right_fit_cr,
            xm_per_pix=xm_per_pix,
            minfot=minfot,
            lane_width_pixel=lane_width_pixel)
    else:
        left_fit, right_fit = prior_left_fit, prior_right_fit
        left_fit_cr, right_fit_cr = prior_left_fit_cr, prior_right_fit_cr

        # timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # dest_folder = "./problem_images"
        # cv2.imwrite(dest_folder + '/' + timestamp + ".jpg", img)

    # Draw the windows on the visualization image
    # Color in left and right line pixels
    draw_on_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    draw_on_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Add lane lines to draw_on_img
    lane_lines = draw_lanes_on_img(
        img=draw_on_img,
        left_fit=left_fit,
        right_fit=right_fit,
        step_size=step_size,
        color=lane_line_color,
        thickness=lane_line_thickness,
        dashed=False)
    draw_on_img = cv2.add(lane_lines, draw_on_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    draw_on_img = cv2.addWeighted(draw_on_img, 1, window_img, 0.3, 0)

    # Get lane curvature
    left_curverad, right_curverad = get_radius_of_curvature(
        ploty=ploty,
        left_fit=left_fit_cr,
        right_fit=right_fit_cr,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix)

    # Get vehicle position wrt lane center
    veh_pos_wrt_ln_ctr = get_vehicle_position_wrt_lane_center(
        imshape=img.shape,
        left_fit=left_fit,
        right_fit=right_fit,
        left_curverad=left_curverad,
        right_curverad=right_curverad,
        xm_per_pix=xm_per_pix)

    return left_fit, right_fit, left_fit_cr, right_fit_cr, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img


def get_radius_of_curvature(ploty, left_fit, right_fit, xm_per_pix, ym_per_pix):
    # Calculate the radius of curvature in meters
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    return left_curverad, right_curverad


def get_calibrated_lane_curves(
        prior_left_fit, prior_right_fit, prior_left_fit_cr, prior_right_fit_cr,
        left_fit, right_fit, left_fit_cr, right_fit_cr,
        xm_per_pix, minfot=0.5, lane_width_pixel=800):
    left_fot = abs(left_fit[1])
    right_fot = abs(right_fit[1])
    # print("left_fot ", left_fot)
    # print("right_fot ", right_fot)

    if left_fot > minfot and right_fot > minfot:
        # Bad curverad on both lines
        return prior_left_fit, prior_right_fit, prior_left_fit_cr, prior_right_fit_cr

    elif left_fot > minfot:
        # Left lane curve is bad
        # print("left lane curve is bad")
        left_fit = np.copy(right_fit)
        left_fit[2] = left_fit[2] - lane_width_pixel
        left_fit_cr = np.copy(right_fit_cr)
        left_fit_cr[2] = left_fit_cr[2] - lane_width_pixel * xm_per_pix
        return left_fit, right_fit, left_fit_cr, right_fit_cr

    elif right_fot > minfot:
        # Right lane curve is bad
        # print("right lane curve is bad")
        right_fit = np.copy(left_fit)
        right_fit[2] = right_fit[2] + lane_width_pixel
        right_fit_cr = np.copy(left_fit_cr)
        right_fit_cr[2] = right_fit_cr[2] - lane_width_pixel * xm_per_pix
        return left_fit, right_fit, left_fit_cr, right_fit_cr

    else:
        return left_fit, right_fit, left_fit_cr, right_fit_cr


def get_vehicle_position_wrt_lane_center(imshape, left_fit, right_fit, left_curverad, right_curverad, xm_per_pix):
    yval = imshape[0]
    img_width = imshape[1]

    # Get offset wrt center in pixel
    left_x = left_fit[0] * yval ** 2 + left_fit[1] * yval + left_fit[2]
    right_x = right_fit[0] * yval ** 2 + right_fit[1] * yval + right_fit[2]
    offset_wrt_center_in_pixel = (left_x + right_x) / 2 - img_width / 2
    offset_wrt_center_in_meter = offset_wrt_center_in_pixel * xm_per_pix

    return offset_wrt_center_in_meter


def highlight_lane_area(img, left_fit, right_fit, y_start=0, y_end=720, highlight_color=[0, 255, 0]):
    hightlight_area_mask = np.zeros_like(img)
    for yval in range(y_start, y_end):
        left_boundary = left_fit[0] * yval ** 2 + left_fit[1] * yval + left_fit[2]
        right_boundary = right_fit[0] * yval ** 2 + right_fit[1] * yval + right_fit[2]
        hightlight_area_mask[yval][int(left_boundary):int(right_boundary)] = highlight_color

    return hightlight_area_mask


def draw_lanes_on_img(img, left_fit, right_fit, step_size=30, color=[255, 0, 0], thickness=10, dashed=False):
    def lane_f_y(yval, lane_fit):
        # Give y, return x = Ay^2 + By + C
        return lane_fit[0] * yval ** 2 + lane_fit[1] * yval + lane_fit[2]

    img_height = img.shape[0]
    pixels_per_step = img_height // step_size
    lane_lines = np.zeros_like(img)

    for i in range(step_size):
        y_start = i * pixels_per_step
        y_end = y_start + pixels_per_step

        left_start_point = (int(lane_f_y(y_start, left_fit)), y_start)
        left_end_point = (int(lane_f_y(y_end, left_fit)), y_end)

        right_start_point = (int(lane_f_y(y_start, right_fit)), y_start)
        right_end_point = (int(lane_f_y(y_end, right_fit)), y_end)

        if dashed == False or i % 2 == 1:
            # Draw left lane segment
            lane_lines = cv2.line(lane_lines, left_end_point, left_start_point, color, thickness)
            # Draw right lane segment
            lane_lines = cv2.line(lane_lines, right_end_point, right_start_point, color, thickness)

    return lane_lines


def add_lane_info_to_image(img, left_curvature, right_curvature, veh_pos_wrt_ln_ctr):
    information_canvas = np.zeros_like(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Add curvature info to image
    cv2.putText(
        information_canvas, 'CurveRad (L): {:.0f}(m)'.format(left_curvature), (50, 50),
        font, 1, (255, 255, 255), 2)
    cv2.putText(
        information_canvas, 'CurveRad (R): {:.0f}(m)'.format(right_curvature), (50, 100),
        font, 1, (255, 255, 255), 2)

    # Add relative vehicle position info to image
    left_or_right = "left" if veh_pos_wrt_ln_ctr > 0 else "right"
    cv2.putText(
        information_canvas,
        'Vehicle is {:.2f}(m) {} of center'.format(np.abs(veh_pos_wrt_ln_ctr), left_or_right), (50, 150),
        font, 1, (255, 255, 255), 2)

    combined_img = cv2.add(information_canvas, img)
    return combined_img


def extract_frames_from_video(clip, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    i = 0
    for frame in clip.iter_frames():
        out_path = os.path.join(dest_folder, '{}.jpg'.format(i))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_path, frame)
        i = i + 1
        print("{}th frame being extracted".format(i))


def add_debug_imgs(img, *debug_imgs):
    scaling_factor = 0.25
    x_offset = int(img.shape[1] / 4 * 3)
    y_offset = int(img.shape[0] / 4 * 0)
    final_img = img
    max_debug_imgs = 2
    for i_debug_img in range(0, max_debug_imgs):
        debug_img = cv2.resize(debug_imgs[i_debug_img], (0, 0), fx=scaling_factor, fy=scaling_factor)
        final_img[y_offset:y_offset + debug_img.shape[0], x_offset:x_offset + debug_img.shape[1]] = debug_img
        y_offset = y_offset + debug_img.shape[0]

    return final_img
