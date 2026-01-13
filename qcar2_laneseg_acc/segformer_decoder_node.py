#!/usr/bin/env python3
"""
SegFormer Decoder Node for Isaac ROS v2.1

This node subscribes to TensorList output from TensorRT inference,
performs argmax to get the dominant class per pixel, and publishes
the segmentation mask as a mono8 image for RViz2 visualization.

Author: hackbrian
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

from sensor_msgs.msg import Image
from isaac_ros_tensor_list_interfaces.msg import TensorList

from cv_bridge import CvBridge


class SegformerDecoderNode(Node):
    """
    Decoder node for SegFormer semantic segmentation.
    
    Subscribes to TensorList containing logits from TensorRT,
    performs vectorized argmax, and publishes segmentation mask.
    """

    # Cityscapes color palette (20 classes - official NVIDIA model)
    CITYSCAPES_PALETTE = np.array([
        [0, 0, 0],        # unlabeled (class 0)
        [128, 64, 128],   # road
        [244, 35, 232],   # sidewalk
        [70, 70, 70],     # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],   # traffic light
        [220, 220, 0],    # traffic sign
        [107, 142, 35],   # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],   # sky
        [220, 20, 60],    # person
        [255, 0, 0],      # rider
        [0, 0, 142],      # car
        [0, 0, 70],       # truck
        [0, 60, 100],     # bus
        [0, 80, 100],     # train
        [0, 0, 230],      # motorcycle
        [119, 11, 32],    # bicycle
    ], dtype=np.uint8)

    def __init__(self):
        super().__init__('segformer_decoder_node')

        # Declare parameters
        self.declare_parameter('input_tensor_topic', '/tensor_rt/tensor_output')
        self.declare_parameter('output_mask_topic', '/segmentation/mask')
        self.declare_parameter('output_colored_topic', '/segmentation/colored_mask')
        self.declare_parameter('num_classes', 19)
        self.declare_parameter('input_height', 224)
        self.declare_parameter('input_width', 224)
        self.declare_parameter('publish_colored', True)

        # Get parameters
        input_topic = self.get_parameter('input_tensor_topic').get_parameter_value().string_value
        output_mask_topic = self.get_parameter('output_mask_topic').get_parameter_value().string_value
        output_colored_topic = self.get_parameter('output_colored_topic').get_parameter_value().string_value
        self.num_classes = self.get_parameter('num_classes').get_parameter_value().integer_value
        self.input_height = self.get_parameter('input_height').get_parameter_value().integer_value
        self.input_width = self.get_parameter('input_width').get_parameter_value().integer_value
        self.publish_colored = self.get_parameter('publish_colored').get_parameter_value().bool_value

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # QoS for NITROS compatibility
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber to TensorList from TensorRT
        self.tensor_sub = self.create_subscription(
            TensorList,
            input_topic,
            self.tensor_callback,
            qos
        )

        # Publishers for segmentation output
        self.mask_pub = self.create_publisher(Image, output_mask_topic, qos)
        
        if self.publish_colored:
            self.colored_pub = self.create_publisher(Image, output_colored_topic, qos)

        self.get_logger().info(f'SegFormer Decoder initialized')
        self.get_logger().info(f'  Input topic: {input_topic}')
        self.get_logger().info(f'  Output mask topic: {output_mask_topic}')
        self.get_logger().info(f'  Supports both pre-argmaxed (1,1,H,W) and logits (1,C,H,W) formats')

    def tensor_callback(self, msg: TensorList):
        """
        Process TensorList containing model output.
        
        Supports multiple formats:
        - HWC pre-argmaxed: (H, W, 1) - official NVIDIA model format
        - NCHW pre-argmaxed: (1, 1, H, W) - trtexec compiled format
        - NCHW logits: (1, num_classes, H, W) - requires argmax
        
        Output: mono8 image with class indices
        """
        if len(msg.tensors) == 0:
            self.get_logger().warn('Received empty TensorList')
            return

        try:
            # Get the output tensor
            tensor = msg.tensors[0]
            
            # Parse tensor dimensions
            dims = tensor.shape.dims
            
            # Convert bytes to numpy array (float32)
            data = np.frombuffer(bytes(tensor.data), dtype=np.float32)
            
            # Determine format and extract mask
            if len(dims) == 3:
                # HWC format: (H, W, 1) - official model
                height, width, channels = dims
                output = data.reshape(height, width, channels)
                mask = output[:, :, 0].astype(np.uint8)
                
            elif len(dims) == 4:
                # NCHW format: (1, C, H, W)
                batch, channels, height, width = dims
                output = data.reshape(batch, channels, height, width)
                
                if channels == 1:
                    # Pre-argmaxed: (1, 1, H, W)
                    mask = output[0, 0].astype(np.uint8)
                else:
                    # Logits: (1, num_classes, H, W) - perform argmax
                    mask = np.argmax(output[0], axis=0).astype(np.uint8)
            else:
                self.get_logger().error(f'Unexpected tensor dimensions: {dims}')
                return

            # Create and publish mono8 mask
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            # Optionally publish colored visualization
            if self.publish_colored:
                colored_mask = self._apply_colormap(mask)
                colored_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding='rgb8')
                colored_msg.header = msg.header
                self.colored_pub.publish(colored_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing tensor: {str(e)}')

    def _apply_colormap(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Cityscapes color palette to segmentation mask.
        
        Args:
            mask: (H, W) array with class indices
            
        Returns:
            (H, W, 3) RGB colored mask
        """
        # Clip to valid class range
        mask_clipped = np.clip(mask, 0, len(self.CITYSCAPES_PALETTE) - 1)
        
        # Vectorized color lookup
        colored = self.CITYSCAPES_PALETTE[mask_clipped]
        
        return colored


def main(args=None):
    # Only initialize if not already initialized (prevents double init from launch files)
    if not rclpy.ok():
        rclpy.init(args=args)
    
    node = SegformerDecoderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only shutdown if we initialized it
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
