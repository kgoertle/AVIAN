# utils/help_text.py

def print_detect_help():
    print("""
Detection Help
=====================

Usage:
    avian detect [sources] [flags]
          
----------------------------------

Source Selection:
    sources=<src1 src2 ...>
        One or more video or camera streams.
        Examples:
            usb0
            usb0 usb1
            video.mp4 other.mov

----------------------------------
          
Flags:
    test
        Use the test model directory (~/.avian/runs/test).
        Helpful for debugging or running detection on small models.

    model
        Set a specific official YOLO model to use for detection.
        This could be used to test camera equipment or run off of the larger COCO or DOTA datasets.
        Examples (variants n, m, x, & l available):
            yolov8
            yolov8-obb
            yolo11
            yolo11-obb
            yolo12
            NOTE: OBB models are unavailable for YOLO12-obb.
            
----------------------------------
          
Examples:
    avian detect usb0
    avian detect sources=usb0 usb1
    avian detect test trailcam.mp4 usb0
    avian model=yolo12m source=usb0

----------------------------------
""")


def print_train_help():
    print("""
Training Help
====================

Usage:
  avian train [options]

----------------------------------
          
Training modes:
    --train, -t
        Transfer-learning mode (default). Loads pretrained weights.
        Might as well specify the model with `model=<name>`!
    
    --scratch, -s
        Train from scratch using an architecture file.
        Might as well specify the architecture with `arch=<name>`!

    update=<run_name> (--update, -u)
        Resume a previous training run ONLY if new images were added.

    resume (--resume, -r)
        Resume from last.pt.

    test (--test, -T)
        Debug mode with reduced settings and output to runs/test/.

----------------------------------
          
Model / Weights:
    model=<name> (--model, -m)
        Use pretrained weights by family:
            yolo11n, yolo11s, yolo11m, yolo11x
            yolov8n, yolov8m, yolov8x
            yolo11-obb-n, etc.

----------------------------------
          
Architecture:
    architecture=<name>, arch=<name>, backbone=<name>, (--arch, -a, -b) 
        Architecture for scratch training.
        Examples:
            yolo11
            yolo11.yaml
            custom_birds.yaml

Dataset selection:
    dataset=<name>, data=<name> (--dataset, -d)
        Choose dataset from ./data/<name>/.

    label-studio=<project> (--labelstudio, -ls)
        Convert a Label Studio export into a YOLO dataset.

----------------------------------
          
Naming:
    name=<run_name> (--name, -n)
      Set output run folder name.

----------------------------------
          
Examples:
    avian train model=yolo11n dataset=birds
    avian train scratch arch=custom.yaml dataset=geckos
    avian train update sparrows
    avian train labelstudio=my_project train

----------------------------------
""")
    
def print_labelstudio_help():
    print("""
Label Studio Help
=====================

Usage:
    avian label-studio [options]

----------------------------------

Purpose:
    Process a Label Studio project into a YOLO-formatted dataset.

    This command looks inside your Label Studio projects folder and:
        - finds a specific project if one is named
        - otherwise uses the most recent unprocessed project
        - converts it into a YOLO dataset inside ./data/

----------------------------------

Project selection:
    label-studio
        Use the most recent unprocessed Label Studio project.

    label-studio=<project>
        Use a specific Label Studio project folder.

    project=<project>
        Alias for label-studio=<project>

----------------------------------

Naming:
    name=<dataset_name> (--name, -n)
        Set a custom name for the processed dataset folder.

        If not provided, the dataset folder will default to a timestamp.

----------------------------------

Examples:
    avian label-studio
    avian label-studio=example
    avian label-studio name=birds_dataset
    avian label-studio=example name=birds_dataset

----------------------------------

Training shortcut:
    Label Studio processing can also be used directly with training:

        avian train label-studio
        avian train label-studio=example
        avian train label-studio=example name=birds_dataset model=yolo11n arch=yolo11n

    When used with training, the processed dataset is automatically used.
    You do not need to specify data=<dataset> separately.

----------------------------------
""")
