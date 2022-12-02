# -*- coding: utf-8 -*-
"""Facial expression recognize

Recognize facial expressions (angry/fear/happy/sad/surprise/neutral) using camera in real-time.

Usage: python camera.py
Make sure the model file (./model/facial_model.h5) 
      and the detector file (./detector/haarcascade_frontalface_alt.xml) exist.
@author:
"""
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
#from keras.preprocessing.image import img_to_array

model_path    = './model/facial_model.h5'
detector_path = './detector/haarcascade_frontalface_alt.xml'

# labels text
labels_list = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']

if __name__ == "__main__":
    # Load model
    model = load_model(model_path)
    detector = cv2.CascadeClassifier(detector_path)
    print('== Model has been loaded successfully. ==')

    # Open camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 800)
    camera.set(4, 800)
    while True:
        t1 = cv2.getTickCount()
        status, frame = camera.read()
        if status:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            cv2.imshow('Facial expression recognize', frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(gray, 1.3, 5)
            if faces is not None:
                for f_x, f_y, f_w, f_h in faces:
                    # Process cropped image
                    face = gray[f_y:f_y+f_h, f_x:f_x+f_w]
                    face = cv2.resize(face, (48, 48))
                    face = np.reshape(face, (1, 48, 48))
                    face = img_to_array(face)
                    face = face / 255.0
                    # Predict
                    Y_pred = model.predict(face)
                    y_pred = y_pred = np.argmax(Y_pred, axis=-1)
                    predict_name = labels_list[y_pred[0]]

                    t2 = cv2.getTickCount()
                    t = (t2 - t1) / cv2.getTickFrequency()
                    fps = 1.0 / t
                    cv2.rectangle(frame, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 3)
                    cv2.putText(frame, '{0}'.format(predict_name), (f_x - 2, f_y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, '{:.2f}'.format(t*1000) + "ms, fps: " + '{:.3f}'.format(fps), 
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Facial expression recognize', frame)
            if cv2.waitKey(10) == ord('q'):
                break
        else:
            break
    # Close camera
    camera.release()
    cv2.destroyAllWindows()