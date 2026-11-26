# Training the board detector and classifier

`board_detector` runs two ONNX models with OpenCV DNN (no PyTorch needed on the Jetson):

| Model | File on the car | Classes | Input |
|---|---|---|---|
| Detector (YOLOv8n) | `~/sar_models/detector.onnx` | `card`, `star`, `cone`, `box` | full frame, 1280 px |
| Classifier (YOLOv8n-cls) | `~/sar_models/classifier.onnx` | `SAFE`, `INJURED`, `HOSTAGE`, `THREAT`, `BACK`, `NOT_TARGET` | board crop, 224 px |

"Detect far, classify close": a 35 cm board is only ~40 px tall at 4 m in HD720, too small to read the
pose, but big enough to *find*. The car then drives to 1.5 m in front of it, where it's ~100 px tall.

## The cards (12 seen so far)

The class comes from the **picture**, not from any colour. The coloured **star** appears only on SAFE cards. It says which exit(s) the car may use: red star → red exit, and so on. Seeing several star colours means any of those exits is allowed.

| Class | Cards | What separates it |
|---|---|---|
| SAFE | 5 × standing man, arm raised: blue star, red star, yellow star, green star, no star | standing, fist up |
| INJURED | cartoon boy crying and holding his knee; photo of a boy holding a red knee | sitting, holding a knee |
| HOSTAGE | cartoon man tied to a chair, tape over his mouth; cartoon man tied to a chair with a gag, sweating; photo of a man tied to a chair, gagged | tied to a chair |
| THREAT | cartoon masked robber holding a man at gunpoint; photo of a man in a balaclava holding a woman at gunpoint | gun + captor |

What this means for training:
- Each class has **both cartoon and photo** versions. Photograph **every** card. A model that only saw the cartoon THREAT may miss the photo THREAT.
- **THREAT also shows a captive person.** Label the whole card THREAT. Never add a separate HOSTAGE box on a THREAT card.
- **Star:** label the star as its own `star` box, on every SAFE card that has one. The node reads its colour (red / green / blue / yellow) from inside that box. A blue shirt on the INJURED photo or a red cone won't count, because only pixels inside a detected star are used.
- The cards are paper wrapped around a box (~30 × 40 cm) standing on the floor, alone or leaning on cones and boxes. The print is on the front only, so also collect **back and side views** for `BACK`.
- Watermarks on some prints (stock-photo text) are fine; they appear in the real cards too.

If you have little time, you can train ONE detector whose classes are the target classes directly
(`SAFE`, `INJURED`, `HOSTAGE`, `THREAT`, `BACK`, `exit_sign`) and set `detector_gives_class: true`.
That's simpler but worse at range.

## 1. Collect images (first lab session)

On the car, with the ZED running:

```bash
ros2 run sar_mission dataset_capture --ros-args -p out_dir:=~/sar_dataset -p period:=0.7
```

Push or drive the car slowly around the boards at the real camera height. Aim for:
- every board type from the front, both sides at ~45 degrees, edge-on and from the back (`BACK`);
- distances from 1 m to 5 m; boards leaning on boxes/cones and partly hidden;
- both lighting conditions in the hall; the exit sign from several distances;
- some frames with **no** board (just obstacles, mats, the star markers) so it learns what isn't a target.

300-600 frames is a good start. Copy them to the laptop: `scp -r nvidia@<car-ip>:~/sar_dataset .`

## 2. Label

Use Roboflow (fastest, free tier), CVAT or labelImg. Export **YOLOv8** format.
- Detector: box every card as `card`, every star as `star`, plus `cone` and `box`.
- Classifier: crop each card box into a folder per class (`SAFE/`, `INJURED/`, ... `BACK/`,
  `NOT_TARGET/` for false boxes). Roboflow can export crops directly ("Classification" export).

Keep 15% of the images for validation, and take them from a **different** recording than training.

## 3. Train (laptop with a GPU, or Google Colab)

```bash
pip install ultralytics
yolo detect train data=sar_det/data.yaml model=yolov8n.pt imgsz=1280 epochs=80 batch=8
yolo classify train data=sar_cls model=yolov8n-cls.pt imgsz=224 epochs=40
```

Check `runs/detect/train/results.png` and the confusion matrix. SAFE vs THREAT and INJURED vs
HOSTAGE are the pairs most likely to be confused; add more examples of whichever pair is weak.

## 4. Export and copy to the car

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=1280 opset=12
yolo export model=runs/classify/train/weights/best.pt format=onnx imgsz=224 opset=12
ssh nvidia@<car-ip> mkdir -p ~/sar_models
scp runs/detect/train/weights/best.onnx  nvidia@<car-ip>:~/sar_models/detector.onnx
scp runs/classify/train/weights/best.onnx nvidia@<car-ip>:~/sar_models/classifier.onnx
```

`opset=12` keeps the model loadable by the older OpenCV on the Jetson. If OpenCV can't load it,
try `opset=11` or `simplify=True`.

## 5. Check on the car

```bash
ros2 launch sar_mission car.launch.py mission:=false
ros2 topic echo /sar/detections        # boxes and probs
ros2 topic hz /sar/detections          # aim for >= 5 Hz; lower imgsz to 960 if it's too slow
```

If it's too slow at 1280 px, export at `imgsz=960` and set `imgsz: 960` in `config/car.yaml`.
