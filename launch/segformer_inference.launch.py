#!/usr/bin/env python3
"""
SegFormer Inference Pipeline Launch File

Launches the complete semantic segmentation pipeline using Isaac ROS v2.1:
1. DNN Image Encoder - Resizes and normalizes input images
2. TensorRT Node - Runs SegFormer inference
3. Decoder Node - Converts logits to segmentation mask

Author: hackbrian
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for SegFormer pipeline."""
    
    # Package share directory
    pkg_share = get_package_share_directory('qcar2_laneseg_acc')
    
    # Launch arguments
    input_image_topic_arg = DeclareLaunchArgument(
        'input_image_topic',
        default_value='/camera/color_image',
        description='Input camera image topic'
    )
    
    # Default paths inside package (official NVIDIA model)
    default_onnx_path = os.path.join(
        pkg_share, 'models', 'oficial', 'citysemsegformer.onnx'
    )
    default_engine_path = os.path.join(
        pkg_share, 'models', 'oficial', 'citysemsegformer.plan'
    )
    
    model_file_path_arg = DeclareLaunchArgument(
        'model_file_path',
        default_value=default_onnx_path,
        description='Path to ONNX model file'
    )
    
    engine_file_path_arg = DeclareLaunchArgument(
        'engine_file_path',
        default_value=default_engine_path,
        description='Path to TensorRT .plan engine file (cache)'
    )
    
    # Get launch configurations
    input_image_topic = LaunchConfiguration('input_image_topic')
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')

    # ==== ComposableNodeContainer with NITROS nodes ====
    # This enables zero-copy data transfer between encoder and TensorRT
    
    nitros_container = ComposableNodeContainer(
        name='segformer_container',
        namespace='segformer',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded container
        composable_node_descriptions=[
            # ---- Node 1: DNN Image Encoder ----
            # Resizes input to 1024x1820, normalizes with custom stats, outputs NCHW tensor
            ComposableNode(
                package='isaac_ros_dnn_image_encoder',
                plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
                name='dnn_image_encoder',
                parameters=[{
                    'input_image_width': 640,
                    'input_image_height': 480,
                    'network_image_width': 1820,
                    'network_image_height': 1024,
                    # Custom normalization from nvinfer_config.txt
                    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
                    'image_mean': [123.675, 116.28, 103.53],
                    'image_stddev': [58.395, 57.12, 57.375],
                    'num_blocks': 40,
                }],
                remappings=[
                    ('image', input_image_topic),
                    ('encoded_tensor', 'tensor_pub'),
                ]
            ),
            
            # ---- Node 2: TensorRT Inference ----
            # Uses pre-compiled .plan engine directly
            ComposableNode(
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensor_rt',
                parameters=[{
                    'model_file_path': model_file_path,      # Static ONNX file
                    'engine_file_path': engine_file_path,    # Generated .plan
                    'input_tensor_names': ['input'],
                    'input_binding_names': ['input'],
                    'output_tensor_names': ['output'],
                    'output_binding_names': ['output'],
                    'verbose': True,
                    'force_engine_update': True,  # Regenerate from static ONNX
                }],
                remappings=[
                    ('tensor_pub', 'tensor_pub'),
                    ('tensor_sub', 'tensor_rt/tensor_output'),
                ],
            ),
        ],
        output='screen',
    )

    # ---- Node 3: SegFormer Decoder (Python) ----
    # Standalone node - cannot be composable (Python)
    # Performs argmax and publishes segmentation mask
    decoder_node = Node(
        package='qcar2_laneseg_acc',
        executable='segformer_decoder_node.py',
        name='segformer_decoder',
        namespace='segformer',
        parameters=[{
            'input_tensor_topic': '/segformer/tensor_rt/tensor_output',
            'output_mask_topic': '/segmentation/mask',
            'output_colored_topic': '/segmentation/colored_mask',
            'num_classes': 20,  # Cityscapes full
            'input_height': 1024,
            'input_width': 1820,
            'publish_colored': True,
        }],
        output='screen',
    )

    return LaunchDescription([
        input_image_topic_arg,
        model_file_path_arg,
        engine_file_path_arg,
        nitros_container,
        decoder_node,
    ])
