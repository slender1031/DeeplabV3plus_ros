# DeeplabV3plus_ros
C++ code for DeeplabV3plus segmentation network (libtorch)

You need to check File **PATH**, **ROS Topics** and the **libtorch version**.


Configuration:
* Ubuntu18.04
* ros-melodic
* OpenCV 3.2.0
* libtorch-cxx11-abi-shared-with-deps-1.7.0
* Pangolin
* Eigen3

Execute the following commands:

```
git clone https://github.com/slender1031/DeeplabV3plus_ros.git
cd deeplab_test_ros
catkin_make
rosrun test_pytorch example-app
```


Segmentation result:

<img src="https://github.com/slender1031/DeeplabV3plus_ros/blob/master/img0000.png" width="720" height="480" />
