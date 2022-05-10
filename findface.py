import cv2
import sys
import os
import logging as log
import datetime as dt
from time import sleep

cascPath = os.path.join(os.getcwd(), "Face-Detection-TEMP-Cascasde.xml")
if not(os.path.isfile(cascPath)):
    _createCascade()

try:
    videoFileName = sys.argv[1]
except IndexError:
    videoFileName = "video.mp4"

if not(os.path.isfile(videoFileName)):
    pass

faceCascade = cv2.CascadeClassifier(cascPath)

log.basicConfig(filename='Face-Detection-{}.log', level=log.INFO)

video_capture = cv2.VideoCapture()
anterior = 0
fc = 1

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1.5)

    if anterior != len(faces):
        anterior = len(faces)
        if len(faces) > 0:
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
            cv2.imwrite(f'extracted/{fc}.jpg', frame)
            fc += 1

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
