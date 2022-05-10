# OpenCV Video Tools

# `vdetect`

## What is this?

This is a tool made to be used with any video, with any cascade. You can run/test your own cascade or use other, already built cascades on videos

## Installation

```bash
sudo apt update
sudo apt install wget python3 python3-pip -y
python3 -m pip install -U opencv-python coloredlogs
wget "https://raw.githubusercontent.com/hirusha-adi/Face-Detection/main/vdetect.py"
chmod +x vdetect.py
sudo mv vdetect.py /usr/local/bin/vdetect
vdetect help
```

## Help

```
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
```

# `findface`

## What is this?

This is a tool built to identify faces of any given video with using the default "haar casface frontal face"

## Installation

```bash
sudo apt update
sudo apt install wget python3 python3-pip -y
python3 -m pip install -U opencv-python coloredlogs
wget "https://raw.githubusercontent.com/hirusha-adi/Face-Detection/main/findface.py"
chmod +x findface.py
sudo mv findface.py /usr/local/bin/findface
findface help
```

## Help

```
Usage -
    findface <video>

Arguments -
    <video> -
        Video name/path. Any format that is supported by opencv-python by default

Example -
    findface videoName.mp4
```
