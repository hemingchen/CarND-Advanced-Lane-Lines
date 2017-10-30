from moviepy.editor import VideoFileClip

from helper_funcs import extract_frames_from_video

# 1) 41 second in project_video
input_path = "project_video.mp4"
subclip = (41, 42)
dest_folder = "./test_images/project_video_{}_{}".format(subclip[0], subclip[1])
input_clip = VideoFileClip(input_path).subclip(subclip[0], subclip[1])
extract_frames_from_video(clip=input_clip, dest_folder=dest_folder)

# 2) 1 second in challenge_video
input_path = "challenge_video.mp4"
subclip = (0, 1)
dest_folder = "./test_images/challenge_video_{}_{}".format(subclip[0], subclip[1])
input_clip = VideoFileClip(input_path).subclip(subclip[0], subclip[1])
extract_frames_from_video(clip=input_clip, dest_folder=dest_folder)
