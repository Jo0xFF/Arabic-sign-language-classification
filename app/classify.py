from collections import deque
import numpy as np
import pandas as pd
import cv2
from app.preprocessing import check_rotation, correct_rotation

def vod_classify(video_path, model, class_names):
	"""
	A Function to Classify a video of Arabic sign language for each sign will be prompted in the upper right of the video.
	Later those signs gestures will be translated into a more readable way using another helper function to read.

	P.S: Classes names right now in the form as the authors written it.

	Args:
	video_path - A relative path to the video.
	model - the model you want to classify the video with.
	class_names - a list of class names (the same form of structure the dataset subfolders.)

	Returns:
	A list of classes to map later into a more readable way.
	"""

	# Reading the video from path
	vod = cv2.VideoCapture(video_path)

	# initialize writer var for writing on desk 
	writer = None

	# Deque for Prediction flickering problem
	all_preds = deque(maxlen=64)

	# The classified classes to translate later on.
	classified_classes = []

	# check if video requires rotation
	rotateCode = check_rotation(video_path)

	while True:
		# Read the video object.
		ret_flag, frame = vod.read()


		# if frame in vod broken stop loop or reached the end of it.
		if not ret_flag:
			break

		# check if the frame needs to be rotated
		if rotateCode is not None:
			frame = correct_rotation(frame, rotateCode)

		# Grab the Width & Height of frame
		# frame = cv2.flip(frame, 0)
		# frame = cv2.flip(frame, 1)
		H, W = frame.shape[:2]

		# Prepare Conversion of frame
		new_frame = frame.copy()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (64, 64)).astype("float32") 

		# Make predictions
		frame_norm = frame/255.
		probs = model.predict(np.expand_dims(frame_norm, axis=0))[0]

		# Perform Rolling Averaging for predictions
		all_preds.append(probs)
		y_preds = np.array(all_preds).mean(axis=0)
		y_preds = np.argmax(y_preds)
		# print(y_preds)
		label = class_names[y_preds]

		# Add the classified class for Concatination later
		classified_classes.append(label)

		# Prepare the Label
		text = f"Sign Detected: {label}"
		cv2.putText(new_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
			1.5, (238, 255, 0), 3)
		
		# Create classified video
		if writer is None:
			writer = cv2.VideoWriter('app/static/uploads/output.webm', cv2.VideoWriter_fourcc(*"VP90"), 25, (W,H))        
		writer.write(new_frame)
	
	writer.release()
	vod.release()
	return classified_classes


def map_words(letter_list, arabic_dict, arabic_class_mapping):
	possible_words = letter_list
	sent = ""
	letter_check = ""
	correct_words = []

	for letter in possible_words:
		letter = arabic_class_mapping[letter]
		if letter_check == letter:
			# Reset letter holder
			letter_check = letter
			# Skip for conflicted letters
			continue
		# Concat the new letter
		sent+= letter

		# Check if new formed letters really a word ?
		if sent in arabic_dict:
			correct_words.append(sent)
			sent = ""
		# Reset letter for holder
		letter_check = letter

	return " ".join(correct_words)