# LaneSeg-ACC

steps:


/usr/src/tensorrt/bin/trtexec \
  --onnx=lane_unet.onnx \
  --saveEngine=lane_unet.plan \
  --fp16 \
  --verbose

  
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation.git

colcon build --symlink-install --packages-up-to isaac_ros_tensor_rt isaac_ros_dnn_image_encoder isaac_ros_nitros

colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_interfaces qcar2_nodes qcar2_lane_following

