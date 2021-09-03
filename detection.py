from numpy.lib.utils import source
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

model = load_model('model.h5')

face_clf = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


src = cv2.VideoCapture(0)

labels_dict = {0:'MASK', 1:'NO MASK'}

color_dict = {0:(0,255,0), 1:(0,0,255)}

while True:
    ret, img = src.read()

    faces = face_clf.detectMultiScale(img, 1.3, 4)

    for x,y,w,h in faces:
        face_img = img[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (224,224))
        normalized = resized/255.0
        resized = np.expand_dims(resized, 0)
        result = model.predict(resized)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x,y), (x+w, y+h), color_dict[label],2)
        cv2.rectangle(img, (x,y-40), (x+w,y), color_dict[label],-1)

        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if(key == 27):
        break


cv2.destroyAllWindows()
src.release()
