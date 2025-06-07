import cv2, pickle
import numpy as np
import tensorflow as tf
# from cnn_tf import cnn_model_fn
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	
	print("prob is=", pred_probab)
	print("class is=", pred_class)
	if pred_probab*100 > 50:
		text = get_pred_text_from_db(pred_class)
	return text

x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)

	img1 = img.copy()
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	threshold_value = 150
	blurred_image = cv2.GaussianBlur(img1, (11, 11), 0)
	_, thresh = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	thresh = thresh[y:y+h, x:x+w]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = cv2.bitwise_not(thresh)
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		# print("len of contours=", len(contours))
		# if len(contours)>29:
		# 	print("I live across street")
		# elif len(contours)>22:
		# 	print("how r u today")
		# elif(len(contours))>20:
		# 	print("u look great")
		# else:
		# 	print("i know right")
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 20:
					if len(text) == 1:
						Thread(target=say_text, args=(text, )).start()
					word = word + text
					count_same_frame = 0

			elif cv2.contourArea(contour) < 1000:
				if word != '':
					Thread(target=say_text, args=(word, )).start()
				text = ""
				word = ""
		else:
			if word != '':
				Thread(target=say_text, args=(word, )).start()
			text = ""
			word = ""
		print("text is:", text)
		# blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		# cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
		# cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
		# cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		# if is_voice_on:
		# 	cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		# else:
		# 	cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		# res = np.hstack((img, blackboard))
		# cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	# if keypress == ord('c'):
	# 	return 2
	# else:
	# 	return 0

def recognize():
	cam = cv2.VideoCapture(0)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		else:
			break

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		
recognize()
