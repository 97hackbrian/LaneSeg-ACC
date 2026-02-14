# LaneSeg-ACC

<<<<<<< HEAD
Proyecto orientado a la segmentación y detección de carriles (lane detection / lane segmentation) para aplicaciones de conducción autónoma y robótica móvil.

## Enlaces de interés y referencias

### Sistemas y pipelines completos
- **Self-driving-ish Computer Vision System**  
  https://github.com/iwatake2222/self-driving-ish_computer_vision_system

### Modelos y métodos de detección de carriles
- **Ultra-Fast Lane Detection**  
  https://github.com/cfzd/Ultra-Fast-Lane-Detection

- **Lane Detection con U-Net**  
  https://github.com/AnshChoudhary/Lane-Detection-UNet/tree/main

- **PriorLane**  
  https://github.com/vincentqqb/PriorLane

- **PersFormer – 3D Lane Detection**  
  https://github.com/OpenDriveLab/PersFormer_3DLane

### Transformers y segmentación semántica
- **Fine-Tuning SegFormer para Lane Detection (BDD100K)**  
  https://github.com/spmallick/learnopencv/blob/master/Fine-Tuning-SegFormer-For-Lane-Detection/Fine-Tune-SegFormer-BDD.ipynb

- **SegFormer preentrenado (Cityscapes – NVIDIA TAO)**  
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/pretrained_segformer_cityscapes?version=deployable_fan_tiny_hybrid_v1.0

### ROS 2 y NVIDIA Isaac
- **Isaac ROS – Image Segmentation**  
  https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_segmentation

### Colecciones y listas curadas
- **Awesome Lane Detection**  
  https://github.com/Core9724/Awesome-Lane-Detection

- **Lane Detection – Awesome List**  
  https://github.com/xukun-zhang/Lane-Detection-awesome-lane-detection



  Propuesta 1:
  
```mermaid
---
config:
  look: neo
  theme: neutral
  layout: dagre
---
flowchart LR
 subgraph HW_IN["Input Hardware"]
    direction TB
        Lidar["2D Lidar"]
        RGB["RGBD Image"]
        IMU["IMU"]
  end
 subgraph L4["Level 4: Security"]
        Safety("Safety Guardian <br>(Emergency Brake)")
  end
 subgraph L3["Level 3: Hybrid Perception"]
        Vision("Vision Detection model<br>")
        RoadSeg("Road Segmentation model<br>")
        SLAM("AMCL<br>(localization)")
  end
 subgraph L1["Level 1: Brain &amp; Var"]
        Orchestrator@{ label: "Orchestrator<br><span style=\"color:\">(Behavior Tree)</span>" }
        Taxi[("Planner Server<br>(Route Map)")]
        Eval[("Logs/Rosbag")]
  end
 subgraph L2["Level 2: Control"]
        GlobalPlan("Planner Server<br>(Route Map)")
        LocalControl("Hybrid Controller<br>(MPPI + Visual)")
        Mux("Cmd_Vel Mux<br>(prioritizer)")
  end
 subgraph SYSTEM["LEVEL4 AUTONOMOUS"]
    direction TB
        L4
        L3
        L1
        L2
  end
 subgraph HW_OUT["Output Hardware"]
        Motors("Motors / Servo / Leds")
  end
    Vision ~~~ RoadSeg
    Lidar ~~~ RGB
    RGB ~~~ IMU
    Lidar -- Range --> Safety
    Lidar --> SLAM
    RGB -- Image --> Vision
    RGB -- Depth / Image --> RoadSeg
    IMU --> SLAM & LocalControl
    Vision -- Objects --> LocalControl
    SLAM -- Pose --> Orchestrator & n2["VISLAM<br>"]
    Taxi -- goal --> Orchestrator
    Orchestrator -- Nav Actions --> GlobalPlan
    GlobalPlan -- Global Path --> LocalControl
    LocalControl -- "Vel. Nav" --> Mux
    Safety -- STOP --> Mux
    Mux ==> Motors
    Orchestrator -.-> Eval
    Vision -- Events --> Orchestrator
    LocalControl -.-> Eval
    n1["Tracking control MPPI<br>"] -- Tracking_vel --> LocalControl
    RoadSeg -- Visual Path --> n1
    RoadSeg -- 3D Path --> n2
    n2 -- RoadMap --> GlobalPlan

    Orchestrator@{ shape: hexagon}
    n2@{ shape: rounded}
    n1@{ shape: rounded}
     Lidar:::sensor
     RGB:::sensor
     IMU:::sensor
     Safety:::critical
     Vision:::logic
     RoadSeg:::alpamayo
     SLAM:::logic
     Orchestrator:::logic
     Taxi:::logic
     Eval:::logic
     GlobalPlan:::logic
     LocalControl:::logic
     Mux:::hardware
     Motors:::hardware
     n2:::alpamayo
     n1:::alpamayo
    classDef critical fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    classDef logic fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef hardware fill:#eeeeee,stroke:#333,stroke-width:1px
    classDef sensor fill:#fff9c4,stroke:#fbc02d,stroke-width:1px
    classDef alpamayo fill:#dcedc8,stroke:#33691e,stroke-width:2px
    style Orchestrator fill:#FF6D00
    style L4 stroke:#FFCDD2
    style L1 stroke:#AA00FF
    style Motors stroke:#C8E6C9
    style HW_IN stroke:#FFE0B2,fill:transparent
    style HW_OUT stroke:#FF6D00
    style SYSTEM stroke:#FFD600,color:#000000
    linkStyle 3 stroke:#D50000,fill:none
    linkStyle 4 stroke:#616161,fill:none
    linkStyle 5 stroke:#2962FF,fill:none
    linkStyle 6 stroke:#00C853,fill:none
    linkStyle 7 stroke:#757575,fill:none
    linkStyle 8 stroke:#757575,fill:none
    linkStyle 9 stroke:#2962FF,fill:none
    linkStyle 10 stroke:#616161,fill:none
    linkStyle 11 stroke:#616161,fill:none
    linkStyle 13 stroke:#616161,fill:none
    linkStyle 14 stroke:#616161,fill:none
    linkStyle 15 stroke:#000000,fill:none
    linkStyle 16 stroke:#D50000,fill:none
    linkStyle 17 stroke:#000000,fill:none
    linkStyle 19 stroke:#2962FF,fill:none
    linkStyle 21 stroke:#00C853,fill:none
    linkStyle 22 stroke:#00C853,fill:none
    linkStyle 23 stroke:#00C853,fill:none
    linkStyle 24 stroke:#00C853,fill:none

    L_IMU_LocalControl_0@{ curve: natural }
```

=======
steps:

## Instalar cupy para cuda 11

  ```bash
  pip install cupy-cuda11x
  ```
Test:
  ```bash
  python3 utils/test_cupy.py
  ```


## Compile Isaac ros2.1 from source using colcon

  ```bash
    
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

  ```
>>>>>>> origin/image_processing
