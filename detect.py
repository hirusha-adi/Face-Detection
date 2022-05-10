import cv2
import sys
import os
import logging
import coloredlogs
from time import sleep

try:
    videoFileName = sys.argv[1]
except IndexError:
    videoFileName = os.path.join(os.getcwd(), "video.mp4")


if not(os.path.isfile(videoFileName)):
    all_mp4_files_cwd = [x for x in os.listdir(
        os.getcwd()) if x.endswith(".mp4")]
    if len(all_mp4_files_cwd) == 1:
        videoFileName = all_mp4_files_cwd[0]

videoFileNameOnly = videoFileName.split(".")[-2].split("/")[-1]
logFileName = f'Face-Detection-{videoFileNameOnly}.log'
if os.path.isfile(logFileName):
    os.remove(logFileName)

logger = logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(logFileName),
        logging.StreamHandler()
    ],
)
coloredlogs.install(level=logging.DEBUG, logger=logger)

outputFramesFolder = os.path.join(os.getcwd(), "faces")
if not(os.path.isdir(outputFramesFolder)):
    logging.debug("created ./faces directory")
    os.makedirs(outputFramesFolder)
logging.info("images will be saved to ./faces directory")


cascPath = os.path.join(os.getcwd(), "Face-Detection-TEMP-Cascasde.xml")
if not(os.path.isfile(cascPath)):
    logging.error(f"Cascade file not found at {cascPath}")
    sleep(3)
    sys.exit()

faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(videoFileName)
anterior = 0
fc = 1
no = 1

while True:
    if not video_capture.isOpened():
        logging.error("Unable to load the video")
        sleep(3)
        sys.exit()

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
            logging.info(f'found {len(faces)} face at frame {no}')
            cv2.imwrite(f'{outputFramesFolder}/{fc}.jpg', frame)
            fc += len(faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow(f'{videoFileNameOnly}', frame)

    no += 1


video_capture.release()
cv2.destroyAllWindows()
