# MobileNetV1 Converter

Converts TensorFlow checkpoint to Lightspeeur MobileNetV1 format

### Install Dependencies
```
pip install tensorflow wget
pip install git+https://github.com/TeamDollus/lightspeeur.git
```

### Run the Script
```python
python main.py
```

### Result Hierarchy
```
- checkpoints
    - mobilenet_v1_1.0_224.ckpt.data-00000-of-00001
    - mobilenet_v1_1.0_224.ckpt.index
    - mobilenet_v1_1.0_224.ckpt.meta
    - mobilenet_v1_1.0_224.tflite
    - mobilenet_v1_1.0_224_eval.pbtxt
    - mobilenet_v1_1.0_224_frozen.pb
    - mobilenet_v1_1.0_224_info.txt
- mobilenet_v1_1.0_224.tgz
- mobilenet_v1_1.0_224_lightspeeur.h5
```
`mobilenet_v1_1.0_224_lightspeeur.h5` is the final result.