# ncnn
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/Tencent/ncnn/master/LICENSE.txt) 
[![Build Status](https://travis-ci.org/Tencent/ncnn.svg?branch=master)](https://travis-ci.org/Tencent/ncnn)
[![Coverage Status](https://coveralls.io/repos/github/Tencent/ncnn/badge.svg?branch=master)](https://coveralls.io/github/Tencent/ncnn?branch=master)


ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。ncnn 从设计之初深刻考虑手机端的部署和使用。无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行，开发出人工智能 APP，将 AI 带到你的指尖。ncnn 目前已在腾讯多款应用中使用，如 QQ，Qzone，微信，天天P图等。

---

### Support most commonly used CNN network
### 支持大部分常用的 CNN 网络

* Classical CNN: VGG AlexNet GoogleNet Inception ...
* Practical CNN: ResNet DenseNet SENet FPN ...
* Light-weight CNN: SqueezeNet MobileNetV1/V2 ShuffleNetV1/V2 MNasNet ...
* Detection: MTCNN facedetection ...
* Detection: VGG-SSD MobileNet-SSD SqueezeNet-SSD MobileNetV2-SSDLite ...
* Detection: Faster-RCNN R-FCN ...
* Detection: YOLOV2 YOLOV3 MobileNet-YOLOV3 ...
* Segmentation: FCN PSPNet ...

---

### HowTo

**[how to build ncnn library](https://github.com/Tencent/ncnn/wiki/how-to-build) on Linux / Windows / Raspberry Pi3 / Android / NVIDIA Jetson / iOS**

* [Build for NVIDIA Jetson](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-nvidia-jetson)
* [Build for Linux x86](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux-x86)
* [Build for Windows x64 using VS2017](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-windows-x64-using-visual-studio-community-2017)
* [Build for MacOSX](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-macosx)
* [Build for Raspberry Pi 3](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-raspberry-pi-3)
* [Build for ARM Cortex-A family with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
* [Build for iOS on MacOSX with xcode](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macosx-with-xcode)
* [Build for iOS on Linux with cctools-port](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-linux-with-cctools-port)
* [Build for Hisilicon platform with cross-compiling](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-hisilicon-platform-with-cross-compiling)

![](https://raw.githubusercontent.com/biubug6/ncnn_faceboxes/master/model/test.jpg)

