import ffmpeg
import cv2
import os

def check_rotation(path_video_file):
	"""
	A Function to check the oreintation of the video we want to read and process.

	Required to run: installing `ffmpeg-python` from pip,
	Install it using `pip install ffmpeg-python`

	Args:
	path_video_file - A string that indicate the relative path to video in your folder.

	Returns:
	A Correct oreintation of video.
	"""
	if os.name == "nt":
		path_video_file = r"app/static/uploads/firstvod.mp4"

	# this returns meta-data of the video file in form of a dictionary
	meta_dict = ffmpeg.probe(path_video_file)

	# from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
	# we are looking for
	rotateCode = None
	try:
		current_rotation = int(meta_dict['streams'][0]['tags']['rotate'])
		if  current_rotation == 90:
			# rotateCode = cv2.ROTATE_90_CLOCKWISE
			rotateCode = cv2.ROTATE_180
		elif current_rotation == 180:
			# rotateCode = cv2.ROTATE_180
			rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
		elif current_rotation == 270:
			rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
	except:
		return rotateCode

	return rotateCode

def correct_rotation(frame, rotateCode):
	"""
	A Helper function to get the correct oreintation of video file.

	Args:
	frame - A single image from video which parsed frame by frame by another function.
	rotateCode - A rotation of type `int` for video.

	Returns:
	A Rotated frame in the correct direction.
	"""
	return cv2.rotate(frame, rotateCode)
