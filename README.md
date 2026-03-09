# TREVOR
TRansformer Enhanced Vision Object Recognition

ByteTrack implementation for person tracking using PyTorch

### Installation

```
conda create -n ByteTrack python=3.8
conda activate ByteTrack
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install scipy
```

### Note

* The default person detector is `YOLOv8-nano`

### Test

* Configure your video path in `main.py` for testing
* Run `python main.py` for testing

### Results

![Alt Text](./demo/demo.gif)

### Reference

* https://github.com/ultralytics/ultralytics
* https://github.com/jahongir7174/YOLOv8-pt


# Run Project:
Start database
`uvicorn main:app --host 127.0.0.1 --port 8000`

Start camera demo
`python main5.py`

Start Dashboard
`cd demo`
`python main.py`
