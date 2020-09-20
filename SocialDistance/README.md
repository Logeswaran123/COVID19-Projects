# Social Distance

The project main idea is to ensure social distance practice is correctly followed in crowded public places.

### Previous approach shortcoming

Images and video frames are 2D. If we detect a person and calculate the distance between them, the distance might be less even when the persons are far away.
This might be due to the fact that persons are one behind the other and the depth information is not obtained.

### How to account

This project tackles the above mentioned problem by,

1. Finding the depth with [High Quality Monocular Depth Estimation](https://github.com/nianticlabs/monodepth2/). The model is trained on KITTI Dataset.
The output image will contain the depth information i.e. The persons at same depth will have closer pixel intensity than the person far away. The is obtained based on the distance from camera.

2. Detect the person using any person detection model and get the bounding box coordinates.

3. For every two bounding box, calculate the absolute difference between pixels. If the difference is less than the set depth threshold value, the two boxes are in the same depth.

4. Now, calculate the Euclidean distance between the two boxes. If the distance is less than the set distance threshold, the two people are close together. It can be concluded that NO SOCIAL DISTANCE is being followed.

### How to run

1. Download the pretrained model or train own model by refering to [nianticlabs repo](https://github.com/nianticlabs/monodepth2/).

2. 
