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
                    ('image', '/buffer/image_resized'),
                    ('resize/image', '/image_resized'),
                    ('camera_info', '/buffer/camera_info')
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
                    'image_mean': [0.0, 0.0, 0.0],
                    'image_stddev': [1.0, 1.0, 1.0],
                    'tensor_output_order': 'NHWC',
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
    # ---------------------------------------------------------
    # CONTENEDOR 3: INFERENCIA (TensorRT + Decoder)
    # ---------------------------------------------------------
    # NOTA: Asegúrate de tener el archivo .onnx en esta ruta
    # Si solo tienes el .plan viejo, BÓRRALO. Necesitamos regenerarlo.
    MODEL_SOURCE_PATH = '/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.onnx'
    ENGINE_OUTPUT_PATH = '/workspaces/isaac_ros-dev/ros2/src/qcar2_LaneSeg-ACC/models/unet/lane_unet.plan'

    inference_container = ComposableNodeContainer(
        name='inference_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # 1. TensorRT Node (El Motor)
            ComposableNode(
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensor_rt',
                parameters=[{
                    # --- GENERACIÓN AUTOMÁTICA DEL MOTOR ---
                    'model_file_path': MODEL_SOURCE_PATH,   # Le damos el ONNX original
                    'engine_file_path': ENGINE_OUTPUT_PATH, # Donde guardará el .plan nuevo
                    'force_engine_update': True,           # Cambia a True si actualizas el ONNX
                    # ---------------------------------------
                    'input_tensor_names': ['input_tensor'],
                    'output_tensor_names': [OUTPUT_TENSOR],
                    'input_binding_names': [INPUT_TENSOR],
                    'output_binding_names': [OUTPUT_TENSOR],
                    'verbose': True 
                }],
                remappings=[
                    ('tensor_pub', '/tensor_input'),
                    ('tensor_sub', '/tensor_raw_output')
                ]
            ),
        ],
        output='screen'
    )

    # 4. CONTAINER DECODER (Solo la visualización)
    decoder_container = ComposableNodeContainer(
        name='decoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='isaac_ros_unet',
                plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
                name='unet_decoder',
                parameters=[{
                    'network_output_type': 'sigmoid',
                    'color_segmentation_mask_encoding': 'rgb8',
                    'mask_width': NETWORK_WIDTH,
                    'mask_height': NETWORK_HEIGHT,
                    'color_palette': [0x000000, 0x00FF00] 
                }],
                remappings=[
                    ('tensor_sub', '/tensor_decoder_input'), # ENTRADA DESDE EL PYTHON
                    ('unet/raw_segmentation_mask', '/lane_detection/mask_raw'),
                    ('unet/colored_segmentation_mask', '/lane_detection/mask_viz')
                ]
            )
        ],
        output='screen'
    )

    return launch.LaunchDescription([resize_container, encoder_container, inference_container, decoder_container])