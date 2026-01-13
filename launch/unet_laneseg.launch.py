import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    
    # ---------------------------------------------------------
    # 1. CONFIGURACIÓN CORRECTA
    # ---------------------------------------------------------
    # ¡CRÍTICO! Debe coincidir con tu 'ros2 topic echo' (VGA)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Modelo
    INPUT_TENSOR = 'input_0'
    OUTPUT_TENSOR = 'output_0'
    ENGINE_PATH = '/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.plan'

    # ---------------------------------------------------------
    # 2. CONTENEDOR DE NODOS
    # ---------------------------------------------------------
    lane_seg_container = ComposableNodeContainer(
        name='lane_seg_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            
            # --- NODO ENCODER ---
            ComposableNode(
                package='isaac_ros_dnn_image_encoder',
                plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
                name='dnn_image_encoder',
                parameters=[{
                    'input_image_width': CAMERA_WIDTH,
                    'input_image_height': CAMERA_HEIGHT,
                    'network_image_width': 256,
                    'network_image_height': 256,
                    'image_mean': [0.485, 0.456, 0.406],
                    'image_stddev': [0.229, 0.224, 0.225],
                    'enable_padding': False, 
                    'tensor_output_order': 'NCHW' # Vital para PyTorch
                }],
                remappings=[
                    ('image', '/camera/color_image'),
                    ('encoded_tensor', '/tensor_input')
                ]
            ),

            # --- NODO TENSORRT ---
            ComposableNode(
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensor_rt',
                parameters=[{
                    'engine_file_path': ENGINE_PATH,
                    # NO INCLUIMOS model_file_path PARA EVITAR REBUILD/CRASH
                    
                    'input_tensor_names': [INPUT_TENSOR],
                    'output_tensor_names': [OUTPUT_TENSOR],
                    'input_binding_names': [INPUT_TENSOR],
                    'output_binding_names': [OUTPUT_TENSOR],
                    
                    'verbose': True,
                    'force_engine_update': False
                }],
                remappings=[
                    ('tensor_sub', '/tensor_input'),
                    ('tensor_pub', '/tensor_output')
                ]
            ),

            # --- NODO DECODER ---
            ComposableNode(
                package='isaac_ros_unet',
                plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
                name='unet_decoder',
                parameters=[{
                    'network_output_type': 'sigmoid',
                    'color_segmentation_mask_encoding': 'rgb8',
                    'mask_width': 256,
                    'mask_height': 256,
                    'color_palette': [0, 0, 0, 0, 255, 0] # Verde
                }],
                remappings=[
                    ('tensor_sub', '/tensor_output'),
                    ('unet/raw_segmentation_mask', '/lane_detection/mask')
                ]
            )
        ],
        output='screen'
    )

    return launch.LaunchDescription([lane_seg_container])