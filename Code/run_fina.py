import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import cv2
import numpy as np
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh])) 

actions = np.array(['A','B','C','D','E','F'])

no_sequences = 30

sequence_length = 30

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def say_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def keras_process_image(img):
    # print("Original image shape:", img.shape)
    img = cv2.resize(img, (image_x, image_y))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("Resized image shape:", img.shape)
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_from_contour(frame):
    cropframe=frame[40:400,0:300]
    pred_probab, pred_class = keras_predict(model, cropframe)
    return pred_probab

sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.8

def vision():
    sequence = []
    predictions = []
    sentence = []
    threshold = 0.8
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (640,480))
            cropframe = frame[40:400, 0:300]
            image, results = mediapipe_detection(cropframe, hands)
            keypoints = extract_keypoints(results)
            
            # If hands are detected
            if results.multi_hand_landmarks:
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                # Predict only when sequence length is 30
                if len(sequence) == 30:
                    res = get_pred_from_contour(frame)
                    predictions.append(np.argmax(res))
                    # print(res)
                    
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res > threshold: 
                            acc = res * 100
                            # print("Accuracy:", acc)
                            if(acc>99.9):
                                print("Good to see you")
                                text = "Good to see you"
                                say_text(text)
                                
                            elif(acc>96.0):
                                print("Hey! How are you")
                                text = "A medium cup of tea please"
                                say_text(text)
                            
                            else:
                                print("Could you help me cross the road please?")
                                text = "Could you help me cross the road please?"
                                say_text(text)
            
            cv2.imshow("crop", cropframe)
            cv2.waitKey(1)
            
        cap.release()


vision()
 
     