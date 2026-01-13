import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    
    # --- CONFIGURACIÓN ---
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    NETWORK_WIDTH = 256
    NETWORK_HEIGHT = 256
    
    INPUT_TENSOR = 'input_0'
    OUTPUT_TENSOR = 'output_0'
    ENGINE_PATH = '/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.plan'

    # ---------------------------------------------------------
    # CONTENEDOR 1: SOLO REDIMENSIONADO (El Portero)
    # ---------------------------------------------------------
    resize_container = ComposableNodeContainer(
        name='resize_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', # Estándar
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::ResizeNode',
                name='image_resize',
                parameters=[{
                    'output_width': NETWORK_WIDTH,
                    'output_height': NETWORK_HEIGHT,
                    'num_blocks': 40
                }],
                remappings=[
                    ('image', '/camera/color_image'),
                    ('resize/image', '/image_resized'),
                    ('camera_info', '/camera/camera_info')
                ]
            )
        ],
        output='screen'
    )

    # ---------------------------------------------------------
    # CONTENEDOR 2: SOLO ENCODER (El Traductor)
    # ---------------------------------------------------------
    encoder_container = ComposableNodeContainer(
        name='encoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', # Estándar
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_dnn_image_encoder',
                plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
                name='dnn_image_encoder',
                parameters=[{
                    'input_image_width': NETWORK_WIDTH,
                    'input_image_height': NETWORK_HEIGHT,
                    'network_image_width': NETWORK_WIDTH,
                    'network_image_height': NETWORK_HEIGHT,
                    'image_mean': [0.485, 0.456, 0.406],
                    'image_stddev': [0.229, 0.224, 0.225],
                    'tensor_output_order': 'NCHW',
                    'tensor_name': INPUT_TENSOR, 
                    'num_blocks': 40
                }],
                remappings=[
                    ('image', '/image_resized'),
                    ('encoded_tensor', '/tensor_input')
                ]
            )
        ],
        output='screen'
    )

    # ---------------------------------------------------------
    # CONTENEDOR 3: INFERENCIA (TensorRT + Decoder)
    # ---------------------------------------------------------
    inference_container = ComposableNodeContainer(
        name='inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container', # Estándar
        composable_node_descriptions=[
            # TensorRT
            ComposableNode(
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensor_rt',
                parameters=[{
                    'engine_file_path': ENGINE_PATH,
                    'input_tensor_names': [INPUT_TENSOR],
                    'output_tensor_names': [OUTPUT_TENSOR],
                    'input_binding_names': [INPUT_TENSOR],
                    'output_binding_names': [OUTPUT_TENSOR],
                    'verbose': False,
                    'force_engine_update': False
                }],
                remappings=[
                    ('tensor_sub', '/tensor_input'),
                    ('tensor_pub', '/tensor_output')
                ]
            ),
            # Decoder
            ComposableNode(
                package='isaac_ros_unet',
                plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
                name='unet_decoder',
                parameters=[{
                    'network_output_type': 'sigmoid',
                    'color_segmentation_mask_encoding': 'rgb8',
                    'mask_width': NETWORK_WIDTH,
                    'mask_height': NETWORK_HEIGHT,
                    'color_palette': [0, 0, 0, 0, 255, 0]
                }],
                remappings=[
                    ('tensor_sub', '/tensor_output'),
                    ('unet/raw_segmentation_mask', '/lane_detection/mask')
                ]
            )
        ],
        output='screen'
    )

    return launch.LaunchDescription([resize_container, encoder_container, inference_container])