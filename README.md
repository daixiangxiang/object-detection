# object-detection
include two methods of object detection
## mobilenet yolo
**Pytorch uses mobilenet s to build yOLOV4 target detection platform.**
### Mobilene
![image](https://user-images.githubusercontent.com/83350834/150306252-9837875a-ba44-449e-b492-d8eaae66af5e.png)

MobileNet network has a smaller volume, less computation, higher precision. It has great advantages in lightweight neural networks.

Mobilenet series network can be used for classification, and its main function is feature extraction. We can use Mobilenet series network to replace CSPdarknet53 in YOLOV4 for feature extraction, and enhance feature extraction of the three initial effective feature layers with the same shape. You can then replace the Mobilenet series with yoloV4.
### Yolov4
YOLOv4's contributions are as follows:
- An efficient and powerful target detection model is developed. It enables everyone to train a super fast and accurate target detector using either a 1080 Ti or a 2080 TiGPU.
- The influence of the most advanced bag-of-freebies and bag-of-specials target detection methods in detector training is verified.
- Modified the most advanced method to make it more efficient and suitable for single GPU training, including CBN, PAN, SAM, etc.
