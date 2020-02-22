import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing import image

import cv2
import os

import numpy as np
import operator

path = os.path.dirname(os.path.abspath(__file__))
print(path)
model = load_model(path + "/kaggle_emotions_model.h5")

#emotion predictions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    #y_pos = np.arange(len(objects))
    #plt.bar(y_pos, emotions, align='center', alpha=0.5)
    for ob in range(len(objects[:6])):
        print(objects[ob] + ' percentage: ' + str(emotions[ob]))
    index,max_val = max(enumerate(emotions[:6]), key=operator.itemgetter(1))
    max_percent = round(max_val*100,1)
    return objects[index] + ' ' + str(max_percent) + '%'

face_cascade = cv2.CascadeClassifier(path+'/cascades/data/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #flip frame for mirror like effect
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # draw faces rectange
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h+10, x:x+w]
        roi_color = frame[y:y+h+10, x:x+w]
        #resize frame
        test_frame = cv2.resize(roi_gray,(48,48), interpolation = cv2.INTER_AREA)
        #pass through neural network
        test = image.img_to_array(test_frame)
        test = np.expand_dims(test, axis = 0)
        test /= 255
        custom = model.predict(test)

        emotion = emotion_analysis(custom[0])
        font = cv2.FONT_HERSHEY_PLAIN
        font_color = (255,0,0)

        color = (255,0,0) #BGR 0-255
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h+10
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color,stroke)
        cv2.putText(frame,emotion,(x-10,y-10), font, 3, font_color, 1, cv2.LINE_AA)

        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()