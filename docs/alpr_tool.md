# alpr-tool quick guide

Subcommands:

- `roi`: draw ROI on a live source and save percent values to config.
  ```
  ./alpr-tool roi --source rtsp://... --conf ./config/openalpr.conf.defaults
  ```
  Keys: `R` draw, `S` save, `C` clear (enable_roi=0), `Q/ESC` quit.

- `tune`: adjust lightweight pre-processing (brightness/contrast/gamma/CLAHE/sharpen/denoise) with trackbars.
  ```
  ./alpr-tool tune --source video.mp4 --conf ./config/openalpr.conf.defaults
  ```
  Keys: `SPACE` toggle original/processed, `S` save preproc_*, `C` disable preproc, `Q/ESC` quit.

- `preview`: preview detection/OCR overlays (ROI, bboxes, text, confidence, FPS).
  ```
  ./alpr-tool preview --source 0 --conf ./config/openalpr.conf.defaults
  ```

- `export-yolo`: convert YOLO .pt to ONNX using ultralytics and optionally update the config.
  ```
  ./alpr-tool export-yolo --model model.pt --out model.onnx --imgsz 640 --update-conf --conf ./config/openalpr.conf.defaults
  ```
  If ultralytics is missing: `pip install -U ultralytics`.

Notes:
- Config updates preserve unknown keys; if write fails, a `.new` file is created.
- ROI/preproc settings are percent-based and consumed by the core before YOLO/OCR.
- Preproc is off by default; apply to detector only when `preproc_apply_before_detector=1`.

