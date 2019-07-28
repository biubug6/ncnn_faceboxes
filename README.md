## It is a FaceBoxes project based on ncnn

#  FaceBoxes
FaceBoxes is a cpu real-time face detector with high accuracy

# ncnn
ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. 

# Train model in 
[how to train](https://github.com/zisianw/FaceBoxes.PyTorch)

### Model convert form Pytorch to Onnx to Ncnn

repair some bug in onnx2ncnn

1.onnx2ncnn in ncnn don't support the operation which changes Split to Slice. Modifying the onnx2ncnn on the basis of ncnn in order to support Split to Slice   operator.

 2.onnx2ncnn in ncnn would parse to the wrong parameter in reshape operator if reshape operator is 4 dims(eg:torch.reshape(b, c, h, w) to ncnn reshape). 
 
 3.winograd3x3 conv style has bug, I don't use winograd3x3.

# Compile

1. set opencv path in ./faceboxes/CMakeLists.txt file

2. [how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build)

3. cd ./build/faceboxes/ && ./facebox, test.jpg saved in ./model/test.jpg

# faceboxes refers to the following projects:

1.[ncnn](https://github.com/Tencent/ncnn)

2.[facebox](https://github.com/zisianw/FaceBoxes.PyTorch)


![](https://raw.githubusercontent.com/biubug6/ncnn_faceboxes/master/model/test.jpg)

