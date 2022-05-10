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

if videoFileName.lower() == "help":
    print("""
Easy Video Detection - vdetect

Usage -
    vdetect <video> <cascade> <save>

Arguments -
    <video>
        Video name/path. Any format that is supported by opencv-python by default
    
    <cascade>
        any opencv cascade to be used for the video
    
    <save>: "yes" | "no"
        save frames to a folder or not

Example -
    vdetect video.mp4 haarcascade_frontalface_default.xml
""")
    sys.exit()

if not(os.path.isfile(videoFileName)):
    all_mp4_files_cwd = [x for x in os.listdir(
        os.getcwd()) if x.endswith(".mp4")]
    if len(all_mp4_files_cwd) == 1:
        videoFileName = all_mp4_files_cwd[0]

videoFileNameOnly = videoFileName.split(".")[-2].split("/")[-1]
logFileName = f'{videoFileNameOnly}.log'
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

try:
    saveFrames = sys.argv[3]
    if saveFrames.lower() in ("yes", "y", "output", "o", "true", "t"):
        saveFrames = True
    else:
        saveFrames = False
except IndexError:
    saveFrames = False

if saveFrames:
    outputFramesFolder = os.path.join(os.getcwd(), videoFileNameOnly)
    if not(os.path.isdir(outputFramesFolder)):
        logging.debug(f"created ./{videoFileNameOnly} directory")
        os.makedirs(outputFramesFolder)
    logging.info(f"images will be saved to ./{videoFileNameOnly} directory")

try:
    cascPath = sys.argv[2]
except IndexError:
    all_xml_files_cwd = [x for x in os.listdir(
        os.getcwd()) if x.endswith(".xml")]
    if len(all_xml_files_cwd) == 1:
        cascPath = all_xml_files_cwd[0]
    else:
        cascPath = "haarcascade_frontalface_default.xml"

if not(os.path.isfile(cascPath)):
    if cascPath == "haarcascade_frontalface_default.xml":
        logging.error(f"Cascade file is not give. Please refer help.")
    else:
        logging.error(f"Cascade file not found at {cascPath}")
    sleep(2)
    sys.exit()

customCascade = cv2.CascadeClassifier(cascPath)
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

    object = customCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in object:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(object):
        anterior = len(object)
        if len(object) > 0:
            logging.info(f'found {len(object)} item at frame {no}')
            if saveFrames:
                cv2.imwrite(f'{outputFramesFolder}/{fc}.jpg', frame)
            fc += len(object)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow(f'{videoFileNameOnly}', frame)

    no += 1


video_capture.release()
cv2.destroyAllWindows()
