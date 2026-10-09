# Auto-labelling and Training a YOLOv8 Detector with Autodistill

Trains a custom object detector **without manual annotation**. Frames are sampled from
video, auto-labelled by **GroundedSAM** from a plain-text description (an *ontology*),
and the resulting dataset trains a **YOLOv8** model. The trained weights are then run on
new clips to locate the target.

## Pipeline

1. **Sample frames** from the source clips at a per-clip stride.
2. **Define an ontology** — map a natural-language description to a class label
   (e.g. a specific vehicle by make/model/colour, or people distinguished by clothing).
3. **Auto-label** every frame with GroundedSAM to produce a YOLO-format dataset.
4. **Train** YOLOv8 on that dataset (optionally resuming from the previous best
   checkpoint for more epochs).
5. **Evaluate** with the generated confusion matrix / results plots.
6. **Predict** on new clips, tuning the confidence threshold per clip.

## Running

Open `autodistill_yolov8_detection.ipynb` in Google Colab with a GPU runtime and run the
cells top to bottom. Point the frame-sampling `SOURCES` and the prediction
`INPUT_VIDEO_PATH` at your own clips, and edit the `CaptionOntology` prompt to describe
your target.

## Stack

Python, Autodistill (GroundedSAM + YOLOv8 backends), Ultralytics YOLOv8, supervision,
OpenCV, PyTorch.

## Notes

Based on Roboflow's autodistill walkthrough, adapted for a final-year project on locating
a target of interest in surveillance footage. The sample paths point at Google Drive
locations used during development. Base weights download automatically on first run.
