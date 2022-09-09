# Important imports
from app import app
from flask import request, render_template, redirect, url_for
from app.classify import vod_classify, map_words
# from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import tensorflow as tf


arabic_letters_map = {"aleff": "ا",
					  "bb": "ب",
					  "taa": "ت",
					  "thaa": "ث",
					  "jeem": "ج",
					  "haa": "ح",
					  "khaa": "خ",
					  "dal": "د",
					  "thal": "ذ",
					  "ra": "ر",
					  "zay": "ز",
					  "seen": "س",
					  "sheen": "ش",
					  "saad": "ص",
					  "dhad": "ض",
					  "ta": "ط",
					  "dha": "ظ",
					  "ain": "ع",
					  "ghain": "غ",
					  "fa": "ف",
					  "gaaf": "ق",
					  "kaaf": "ك",
					  "laam": "ل",
					  "meem": "م",
					  "nun": "ن",
					  "ha": "ه",
					  "waw": "و",
					  "ya": "ئ",
					  "toot": "ة",
					  "al": "ال",
					  "la": "لا",
					  "yaa": "ي"}

class_names = [
	"ain",
	"al",
	"aleff",
	"bb",
	"dal",
	"dha",
	"dhad",
	"fa",
	"gaaf",
	"ghain",
	"ha",
	"haa",
	"jeem",
	"kaaf",
	"khaa",
	"la",
	"laam",
	"meem",
	"nun",
	"ra",
	"saad",
	"seen",
	"sheen",
	"ta",
	"taa",
	"thaa",
	"thal",
	"toot",
	"waw",
	"ya",
	"yaa",
	"zay",
]

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1000 * 1000
ALLOWED_EXTENSIONS = {"mp4", "avi"}

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():

	words = pd.read_csv("app/docs/Clean_tokens.txt", encoding="utf-16", names=["Words"])
	# Execute if request is get
	if request.method == "GET":
		full_filename = ''
		predicted_label = ""
		is_classified = False
		return render_template("index.html", full_filename=full_filename, predicted_label=predicted_label, 
			is_classified=is_classified, init=False)

	# Execute if reuqest is post
	if request.method == "POST":
		option = request.form['options']
		raw_video = request.files['video_upload']
		video_name = raw_video.filename

		# Check if there's video & allowed extension 
		if raw_video and allowed_file(video_name):
			# video_name = secure_filename(video_name)
			raw_video.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], "firstvod.mp4"))

		video_path = os.path.join(app.config['INITIAL_FILE_UPLOADS'], "firstvod.mp4")
		full_filename = "static/uploads/output.webm"

		# Debug
		# print(f"THIS IS THE VIDEO PATH: {video_path}")

		if option == "model-V1":
			model_1 = tf.keras.models.load_model("app/static/models/Arabic-sign-language-translation-CNN")
			classified_letters = vod_classify(video_path=video_path, model=model_1, class_names=class_names)
			sentence = map_words(letter_list=classified_letters, arabic_dict=np.array(words["Words"]), 
								 arabic_class_mapping=arabic_letters_map)

			# Remove the original Video file
			os.remove(video_path)
			# Delete Model to free up memory
			del model_1
			# Turn Flag on for classification success
			is_classified = True

			return render_template("index.html", full_filename=full_filename, predicted_label=sentence,
				is_classified=is_classified, init=True)

		elif option == "model-V2":
			# Load the model V2
			model_2 = tf.keras.models.load_model("app/static/models/Arabic-sign-language-translation-CNN-3RD-EDITON")
			classified_letters = vod_classify(video_path=video_path, model=model_2, class_names=class_names)
			sentence = map_words(letter_list=classified_letters, arabic_dict=np.array(words["Words"]), 
								 arabic_class_mapping=arabic_letters_map)
			
			# Remove the original Video file
			os.remove(video_path)
			# Delete Model to free up memory
			del model_2
			# Turn Flag on for classification success
			is_classified = True

			return render_template("index.html", full_filename=full_filename, predicted_label=sentence,
				is_classified=is_classified, init=True)


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Main function
if __name__ == '__main__':
	app.run(debug=True)
