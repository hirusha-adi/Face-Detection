import cv2
import sys
import os
import logging
import coloredlogs
import datetime as dt
from time import sleep


def _createCascade():
    pass


cascPath = os.path.join(os.getcwd(), "Face-Detection-TEMP-Cascasde.xml")
if not(os.path.isfile(cascPath)):
    _createCascade()

try:
    videoFileName = sys.argv[1]
except IndexError:
    videoFileName = "video.mp4"

if not(os.path.isfile(videoFileName)):
    all_mp4_files_cwd = [x for x in os.listdir(
        os.getcwd()) if x.endswith(".mp4")]
    if len(all_mp4_files_cwd) == 1:
        videoFileName = all_mp4_files_cwd[0]

faceCascade = cv2.CascadeClassifier(cascPath)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            f'Face-Detection-{videoFileName.split(".")[-2]}.log'),
        logging.StreamHandler()
    ]
)

video_capture = cv2.VideoCapture(videoFileName)
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        if len(faces) > 0:
            logging.info("faces: "+str(len(faces)) +
                         " at "+str(dt.datetime.now()))
            cv2.imwrite(f'extracted/{fc}.jpg', frame)
            fc += 1

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
