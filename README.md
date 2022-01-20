# object-detection
*include two methods of object detection*
## Faster RCNN
![image](https://user-images.githubusercontent.com/83350834/150303820-85928758-29a3-4315-97a3-bcf193e75448.png)
![image](https://user-images.githubusercontent.com/83350834/150303898-df9e5334-618c-4eeb-902b-d4e26ba833f0.png)
Faster RCNN can simply be regarded as the system of "region generation network + FAST RCNN", which replaces Selective Search method in FAST RCNN with region generation network. Faster RCNN focuses on solving three problems in this system:
- How to design area generation networks?
- How to train a LAN?
- How to make the region generation network and fast RCNN network share the feature extraction network?
