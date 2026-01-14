# LaneSeg-ACC

steps:


/usr/src/tensorrt/bin/trtexec \
  --onnx=lane_unet.onnx \
  --saveEngine=lane_unet.plan \
  --fp16 \
  --verbose


/usr/src/tensorrt/bin/trtexec   --onnx=/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.onnx   --saveEngine=/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.plan   --outputIOFormats=fp32:hwc   --verbose

  
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git
git clone -b release-2.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation.git

colcon build --symlink-install --packages-up-to isaac_ros_tensor_rt isaac_ros_dnn_image_encoder isaac_ros_nitros

colcon build --symlink-install --packages-up-to qcar2_autonomy qcar2_interfaces qcar2_nodes qcar2_lane_following


edit file: run_dev.sh

# Run container from image
print_info "Running $CONTAINER_NAME"
docker run -it --rm \
    --privileged \
    --network host \
    ${DOCKER_ARGS[@]} \
    -v $ISAAC_ROS_DEV_DIR:/workspaces/isaac_ros-dev \
    -v /dev/*:/dev/* \
    -v /etc/localtime:/etc/localtime:ro \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=4g \
    --user="admin" \
    --group-add render \
    --device /dev/dri:/dev/dri \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    $@ \
    $BASE_NAME \
    /bin/bash


edit file: Dockerfile.quanser
    #RUN echo "source /workspace/cartographer_ws/install/setup.bash" >> /home/admin/.bashrc