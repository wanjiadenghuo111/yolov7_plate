车牌检测识别onnx导出与推理

1.检测onnx导出

```
python models/export.py --weights weights/yolov7-lite-t.pt --grid

```

2.识别onnx导出看这里 [车牌识别](https://github.com/we0091234/crnn_plate_recognition)
```
python export_lpr.py 
```

3.车牌检测+车牌识别  onnx推理

```
 python onnx/yolov7_plate_onnx_infer.py --detect_model weights/yolov7-lite-s.onnx --rec_model weights/plate_rec_color.onnx --image_path imgs --output result
```

4.车牌检测+车牌识别  onnx推理 c++
```
sudo apt-get install libopencv-dev python3-opencv
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz

 g++ -o yolov7_plate_onnx_infer yolov7_plate_onnx_infer.cpp -Ionnxruntime-linux-x64-gpu-1.22.0/include -I/usr/include/opencv4/ -L/usr/lib/x86_64-linux-gnu/ -Lonnxruntime-linux-x64-gpu-1.22.0/lib/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lonnxruntime

 ./yolov7_plate_onnx_infer
```
