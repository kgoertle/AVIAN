# AVIAN
**Automated Visual Identification & Analysis Network**

An open-source, automated animal-behavior detection pipeline.

## Overview
**AVIAN (1.0.0)** is a research-oriented, OpenCV / Ultralytics-based pipeline designed to make custom deep-learning model training & behavioral detection accessible to field & laboratory researchers.  

**Supports:**

- Multi-source real-time inference (video & live camera feeds).
- Structured logging of detections, interactions, & per-frame aggregate statistics.
- Automatic metadata extraction for precise timestamping for video & camera sources.
- Full configurability & modular design for research reproducibility.

This project remains open-source & under active development that began as an undergraduate research initiative. Contributions & feedback are always welcome!

## Features

### Model Training
- Supports **transfer learning** or **training from scratch** of an existing model.  
- Automatically exports **training metrics** to:
  - `Weights & Biases` (W&B)  
  - `quick-summary.txt` (local lightweight summary)
- Supports **aggressive data augmentation** & **auto-detection of new data** for retraining.

### Detection Pipeline
- **Multi-threaded inference** across multiple sources (camera feeds & videos).  
- **Metadata-aware timestamping** for accurate temporally-aligned measurements.
- **Centralized message handling** using unified `DetectUI` for all info, warnings, errors, & save confirmations.
- **Robust exception handling** for model initialization, frame errors, & I/O failures.

### Classes & Configuration
- The pipeline uses **user-defined class configurations**:
  - `FOCUS_CLASSES`: primary subjects (e.g., animal species)
  - `CONTEXT_CLASSES`: contextual or environmental elements (e.g., feeders, water trays, etc)
- Class lists are stored in & managed through `classes_config.yaml` within the config folder, allowing for easy modification without editing code.


#### Here is an example of a default _classes config YAML_ file:
```
FOCUS_CLASSES:
- F
- M
- Feeder
- Main_Perch
- Nesting_Box
- Sky_Perch
- Wooden_Perch

CONTEXT_CLASSES: []

```

A model's class list is extracted straight from its `model.pt` file to prepare a `class_config.yaml` file, which will be located in the `/configs/<model_name>` path.

This allows for a class list to be divided between `focus` & `context` classes that simplifies output statistics & terminal logs.

This class setup is intended to set specific classes as _objects_ for _focus_ classes to interact with, giving context to those interactions. 
Note, as well, that all classes labeled as `context` are expected to be static objects, and therefore, have their associated bounding boxes _anchored_. This helps stablize overlaps.

Please ensure that the [] are removed if defining `context` classes!

### Measurement System
- Data collection is centralized within a single helper utility that handles:
  - Passive presence tracked by logging current class counts on a user-defined time interval.
  - Interval-level aggregation for logging raw detection tracks within a user-defined time interval.
  - Session summaries 
  - Interaction tracking (focus vs. context classes)
  - Motion 
- Exports structured `.csv` summaries:
  - `counts.csv`, `average_counts.csv`
  - `interval_results.csv`, `session_summary.csv`
  - `interactions.csv`
  - `motion_counts.csv`
  - `motion_intensity.csv`
  - `motion_prevalence.csv`
Supports automatic calculation of ratios (e.g., M:F) & normalized detection rates.

### Directory & Output Structure
Integrates a **clean, timestamped log structure** for both camera feeds & videos:

**Camera sources:**
```
/AVIAN/logs/<model_name>/measurements/camera-feed/<usb>/<system_timestamp>/<measurements>/
├── recordings/
│   └── <usb>.mp4
└── scores/
    ├── <usb>_metadata.jsonw=
    ├── counts/
    │   ├── counts.csv
    │   ├── average_counts.csv
    │   ├── frame_counts.csv
    │   └── session_summary.csv
    ├── interactions/
    │   └── interactions.csv
    └── motion/
        ├── motion_counts.csv
        ├── motion_intensity.csv
        └── motion_prevalence.csv
```

**Video sources:**
```
/AVIAN/logs/(model_name)/measurements/video-in/<video>/<video_timestamp>/measurements/
├── recordings/
│   └── <video>.mp4
└── scores/
    ├── source_metadata.json
    ├── frame-data/
    │   ├── interval_results.csv
    │   └── session_summary.csv
    ├── counts/
    │   ├── counts.csv
    │   └── average_counts.csv
    ├── interactions/
    │   └── interactions.csv
    └── motion/
        ├── motion_counts.csv
        ├── motion_intensity.csv
        └── motion_prevalence.csv
```

- Folder names are **automatically sanitized** to avoid filesystem errors.  
- Each source has its own **isolated measurement subdirectory**.  

### Terminal UI
Note that AVIAN is a _headless_ detection pipeline, meaning that live display windows will _not_ appear while running inference. 
Instead, the terminal logs & tracks initiation, FPS, & basic statistics.

#### Here is an example of what to expect from the terminal:
```
Detection
----------------

model: <model>

<video>: Frames:-- | FPS:-- | Time:-- | ETA:--
  class1:-
  class2:-
  OBJECTS:-

<usb>: Frames:-- | FPS:-- | Time:--
  class1:-
  class2:-
  OBJECTS:-

------------------------------------------------------------------------------------------------

model: <model>

info: Loaded 6 classes: ['class1', 'class2', 'class3', 'class4', 'class5', 'class6']
info: Recording initialized at mm/dd/yyy hh:mm:ss
info: Source '<video>' completed.

------------------------------------------------------------------------------------------------

exit: Stop signal received. Terminating pipeline...
exit: Saving CSV spreadsheets...

------------------------------------------------------------------------------------------------

model: <model>

save: Measurements for <video>
save: Measurements saved to: "measurements/video-in/<video>/<video_timestamp>/scores"
      - <video>.mp4
      - <video>_metadata.json
      - counts.csv
      - average_counts.csv
      - interval_results.csv
      - session_summary.csv
      - interactions.csv
      - motion_counts.csv
      - motion_intensity.csv
      - motion_prevalence.csv

save: Measurements for <usb>
save: Measurements saved to: "measurements/camera-feeds/<usb>/<system_timestamp>/scores"
      - <usb>.mp4
      - <usb>_metadata.json
      - counts.csv
      - average_counts.csv
      - interval_results.csv
      - session_summary.csv
      - interactions.csv
      - motion_counts.csv
      - motion_intensity.csv
      - motion_prevalence.csv

exit: All detection threads safely terminated.
```

## Default example model trained on _7 classes_:
  - `M` (Male Passer domesticus)
  - `F` (Female Passer domesticus)
  - `Feeder`
  - `Main_Perch`
  - `Wooden_Perch`
  - `Sky_Perch`
  - `Nesting_Box`

The model was trained **using this pipeline** & has been used primarily for testing purposes.

The purpose of this model in particular is use for tracking & logging basic behavioral attributes of captive _Passer domesticus_ subjects influenced by various intestinal microbial communities over an individual's development.

To be clear, the model is still in development & included for users to demonstrate running inference using a custom model trained through the pipeline.

## Installation
#### 1. Install MiniConda or Conda:
`https://www.anaconda.com/docs/getting-started/miniconda/main`

`https://www.anaconda.com/download`

#### 2. Create & activate environment using:
`conda create -n AVIAN python=3.10`

`conda activate AVIAN`

#### 3. Install the package:
`pip install avian-cv`

### Prerequisites
- Must use `Python 3.10` or older.
- Keep in mind, training & detection require entirely separate system requirements.
- A computer with a relatively powerful CPU or has a GPU with `CUDA enabled` is required.

## Execution
### Initiate Training
#### - Transfer-learning by default:
`avian train`

**Option to specify weights from either OBB or standard YOLO model:**

`avian train model=(yolo11n, yolo11l-obb, yolov8m, etc.)`

This will **default** to using YOLO11n.pt if not specified.


**Option to name the model:**

`avian train name="my awesome run!!"`

**Option to specify dataset within `data` folder.**

`avian train data="my awesome dataset!!`

This will **default** to the most recent dataset within the /data folder.


#### - Train a model only from custom dataset:

**Option to specify weights from either OBB or standard YOLO model.**

`avian train architecture=(yolo11, yolo12, yolov8-obb, etc.)`

This will **default** to YOLO11.yaml if not specified.


#### - Designed to allow users to debug training operation:
`avian train test`

#### - Process Label-Studio export folders:
`avian labelstudio="my awesome export!!"`

**NOTE: Many of these commands can be set together, so here are a few examples:**

`avian train labelstudio=geckos model=yolo11m architecture=customgeckomodel`

`avian train data=geckos model=yolo12 test`


### Initiate Detection
#### - Defaults to mostly recently trained model & initiates usb0:
`avian detect`

#### - Initiate multiple sources in parallel:
`avian detect usb0 usb1 "video1.type" "video2.type"`

#### - Run inference using an official YOLO model or custom model:
`avian detect model=(yolo11, yolo12, yolov8-obb, etc.)`

#### - Designed to allow users to route to debug model:
`avian detect test`
