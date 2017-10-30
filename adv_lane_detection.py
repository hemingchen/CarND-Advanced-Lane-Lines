# Helper functions

import functools

import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from helper_funcs import *
from param_defs import *


def lane_detection_pipeline(
        raw_img,
        mtx,
        dist,
        ksize,
        xgrad_thresh,
        ygrad_thresh,
        mag_thresh,
        dir_thresh,
        h_thresh,
        l_thresh,
        s_thresh,
        vertices,
        M,
        Minv,
        xm_per_pix,
        ym_per_pix,
        nwindows,
        margin,
        minpix,
        minfot,
        lane_width_pixel,
        step_size,
        lane_color,
        lane_thickness,
        highlight_lane_area_y_start,
        highlight_lane_area_y_end,
        debug):
    global prior_left_fit, prior_right_fit

    # Check raw image dimensions
    img_shape = raw_img.shape

    # Undistort image
    undistorted_raw_img = cv2.undistort(raw_img, mtx, dist, None, mtx)

    # Apply thresholds
    combined_binary = apply_multiple_thresholds(
        img=undistorted_raw_img,
        ksize=ksize,
        xgrad_thresh=xgrad_thresh,
        ygrad_thresh=ygrad_thresh,
        mag_thresh=mag_thresh,
        dir_thresh=dir_thresh,
        h_thresh=h_thresh,
        l_thresh=l_thresh,
        s_thresh=s_thresh)

    # Apply mask
    masked_img = region_of_interest(combined_binary, vertices=vertices)

    # Change perspective
    warped_img = cv2.warpPerspective(masked_img, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR)

    left_fit, right_fit, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img = \
        None, None, None, None, None, None

    # If not prior window information, use sliding window to detect lane lines
    if prior_left_fit is None or prior_right_fit is None:
        # print("use sliding window")

        # Get starting points on the bottom of the image
        leftx_base, rightx_base = get_starting_points_of_lane_lines(img=warped_img)

        left_fit, right_fit, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img = \
            get_lane_lines_with_sliding_window(
                img=warped_img,
                leftx_base=leftx_base,
                rightx_base=rightx_base,
                prior_left_fit=prior_left_fit,
                prior_right_fit=prior_left_fit,
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
    else:
        # print("use prior frame info")
        left_fit, right_fit, left_curverad, right_curverad, veh_pos_wrt_ln_ctr, draw_on_img = \
            get_lane_lines_based_on_prior_frame_lines(
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

    # Update cache
    prior_left_fit = left_fit
    prior_right_fit = right_fit

    # Create an empty canvas
    blank_canvas = np.zeros_like(raw_img)

    # Draw lanes
    lane_img = draw_lanes_on_img(
        img=blank_canvas,
        left_fit=left_fit,
        right_fit=right_fit,
        step_size=step_size,
        color=lane_color,
        thickness=lane_thickness,
        dashed=False)

    # Highlight lane area
    hightlight_area_mask = highlight_lane_area(
        img=blank_canvas,
        left_fit=left_fit,
        right_fit=right_fit,
        y_start=highlight_lane_area_y_start,
        y_end=highlight_lane_area_y_end,
        highlight_color=[0, 255, 0])
    highlighted_lane_img = cv2.add(lane_img, hightlight_area_mask)

    # Change perspective
    lane_lines_driver_perspective = cv2.warpPerspective(highlighted_lane_img, Minv, (img_shape[1], img_shape[0]),
                                                        flags=cv2.INTER_LINEAR)

    # Draw lanes on top of original image
    undistorted_raw_img_with_lanes = cv2.add(lane_lines_driver_perspective, undistorted_raw_img)

    # Draw lane information on image
    undistorted_raw_img_with_lanes_info = add_lane_info_to_image(
        img=undistorted_raw_img_with_lanes,
        left_curvature=left_curverad,
        right_curvature=right_curverad,
        veh_pos_wrt_ln_ctr=veh_pos_wrt_ln_ctr)

    if debug:
        debug_img_1 = np.dstack((masked_img, masked_img, masked_img)) * 255
        debug_img_2 = draw_on_img
        final_debug_img = add_debug_imgs(undistorted_raw_img_with_lanes_info, debug_img_1, debug_img_2)

        return final_debug_img

    else:
        return undistorted_raw_img_with_lanes_info


def test_pipeline_on_one_image(raw_img):
    final_output_1 = lane_detection_pipeline(
        raw_img=raw_img,
        mtx=mtx,
        dist=dist,
        ksize=ksize,
        xgrad_thresh=xgrad_thresh,
        ygrad_thresh=ygrad_thresh,
        mag_thresh=mag_thresh,
        dir_thresh=dir_thresh,
        h_thresh=h_thresh,
        l_thresh=l_thresh,
        s_thresh=s_thresh,
        vertices=vertices,
        M=M,
        Minv=Minv,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix,
        nwindows=nwindows,
        margin=margin,
        minpix=minpix,
        minfot=minfot,
        lane_width_pixel=lane_width_pixel,
        step_size=step_size,
        lane_color=lane_color,
        lane_thickness=lane_thickness,
        highlight_lane_area_y_start=highlight_lane_area_y_start,
        highlight_lane_area_y_end=highlight_lane_area_y_end,
        debug=debug)

    plt.figure(figsize=(10, 5))
    plt.imshow(final_output_1)
    plt.title("Final Output of Lane Detection Pipeline", fontsize=20)
    plt.show()


def test_pipeline_on_video(input_path, output_path, subclip_range=None):
    partial_lane_detection_pipeline = functools.partial(
        lane_detection_pipeline,
        mtx=mtx,
        dist=dist,
        ksize=ksize,
        xgrad_thresh=xgrad_thresh,
        ygrad_thresh=ygrad_thresh,
        mag_thresh=mag_thresh,
        dir_thresh=dir_thresh,
        h_thresh=h_thresh,
        l_thresh=l_thresh,
        s_thresh=s_thresh,
        vertices=vertices,
        M=M,
        Minv=Minv,
        xm_per_pix=xm_per_pix,
        ym_per_pix=ym_per_pix,
        nwindows=nwindows,
        margin=margin,
        minpix=minpix,
        minfot=minfot,
        lane_width_pixel=lane_width_pixel,
        step_size=step_size,
        lane_color=lane_color,
        lane_thickness=lane_thickness,
        highlight_lane_area_y_start=highlight_lane_area_y_start,
        highlight_lane_area_y_end=highlight_lane_area_y_end,
        debug=debug)

    input_clip = VideoFileClip(input_path)
    if subclip_range is not None:
        input_clip = input_clip.subclip(subclip_range[0], subclip_range[1])

    output_clip = input_clip.fl_image(partial_lane_detection_pipeline)
    output_clip.write_videofile(output_path, audio=False)


if __name__ == "__main__":
    # 1) Test pipeline once
    prior_left_fit, prior_right_fit = None, None
    raw_img = mpimg.imread("test_images/challenge_video_0_1/1.jpg")
    test_pipeline_on_one_image(raw_img)

    # 2) Test pipeline on project video
    prior_left_fit, prior_right_fit = None, None
    input_path = "test_videos/project_video.mp4"
    output_path = "output_videos/project_video_output.mp4"
    subclip_range = None # (41, 42)
    test_pipeline_on_video(
        input_path=input_path,
        output_path=output_path,
        subclip_range=subclip_range)

    # 3) Test pipeline on challenge video
    # prior_left_fit, prior_right_fit = None, None
    # input_path = "test_videos/challenge_video.mp4"
    # output_path = "output_videos/challenge_video_output.mp4"
    # subclip_range = None
    # test_pipeline_on_video(
    #     input_path=input_path,
    #     output_path=output_path,
    #     subclip_range=subclip_range)