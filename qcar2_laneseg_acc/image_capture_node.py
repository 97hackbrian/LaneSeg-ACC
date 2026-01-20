#!/usr/bin/env python3
"""
ROS 2 Node for Data Acquisition - Semantic Segmentation Training Dataset
Author: hackbrian
Description: Captures images from camera topic with intelligent throttling for U-Net training
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from pathlib import Path


class ImageCaptureNode(Node):
    """
    ROS 2 node for capturing training images from camera stream.
    
    Features:
    - Subscribes to /camera/color_image
    - Detects and logs image resolution on first frame
    - Throttles capture rate to avoid redundant frames
    - Saves images with timestamp-based naming
    - Robust error handling
    """
    
    def __init__(self):
        super().__init__('image_capture_node')
        
        # Declare parameters
        self.declare_parameter('throttle_interval', 1.0)  # seconds between captures
        self.declare_parameter('output_dir', 'train_unet/training_data/raw_images')
        self.declare_parameter('camera_topic', '/camera/color_image')
        
        # Get parameters
        self.throttle_interval = self.get_parameter('throttle_interval').value
        self.output_dir = self.get_parameter('output_dir').value
        self.camera_topic = self.get_parameter('camera_topic').value
        
        # Initialize variables
        self.bridge = CvBridge()
        self.last_capture_time = 0.0
        self.first_frame = True
        self.image_count = 0
        
        # Create output directory
        self._create_output_directory()
        
        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Image Capture Node initialized')
        self.get_logger().info(f'Subscribing to: {self.camera_topic}')
        self.get_logger().info(f'Throttle interval: {self.throttle_interval}s')
        self.get_logger().info(f'Output directory: {self.output_dir}')
        self.get_logger().info('Waiting for camera images...')
    
    def _create_output_directory(self):
        """Create output directory if it doesn't exist."""
        # Check if output_dir is absolute path
        output_path = Path(self.output_dir)
        
        if output_path.is_absolute():
            # Use absolute path as-is
            full_output_path = output_path
        else:
            # Find package root dynamically by searching for package.xml
            package_root = self._find_package_root()
            if package_root:
                full_output_path = package_root / self.output_dir
                self.get_logger().info(f'Package root detected: {package_root}')
            else:
                # Fallback to current working directory if package.xml not found
                full_output_path = Path.cwd() / self.output_dir
                self.get_logger().warn('Could not detect package root, using current directory')
        
        try:
            full_output_path.mkdir(parents=True, exist_ok=True)
            self.output_path = str(full_output_path)
            self.get_logger().info(f'Output directory ready: {self.output_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to create output directory: {e}')
            raise
    
    def _find_package_root(self):
        """
        Find the package root by searching for the source workspace.
        This works even when the node is executed from the install directory.
        Returns the Path to the package root, or None if not found.
        """
        # Strategy 1: Search upward from __file__ for package.xml (works in source)
        current_path = Path(__file__).resolve().parent
        for _ in range(10):
            package_xml = current_path / 'package.xml'
            if package_xml.exists():
                self.get_logger().info(f'Found package.xml at: {current_path}')
                return current_path
            
            parent = current_path.parent
            if parent == current_path:
                break
            current_path = parent
        
        # Strategy 2: If running from install, search for src/qcar2_laneseg_acc
        # Typical install path: /path/to/ros2/install/qcar2_laneseg_acc/lib/qcar2_laneseg_acc/
        # We need to go to: /path/to/ros2/src/qcar2_LaneSeg-ACC/
        current_path = Path(__file__).resolve()
        
        # Search for 'install' in the path
        parts = current_path.parts
        if 'install' in parts:
            install_idx = parts.index('install')
            # Workspace root is parent of 'install'
            workspace_root = Path(*parts[:install_idx])
            
            # Try common source directory patterns
            src_patterns = [
                workspace_root / 'src' / 'qcar2_LaneSeg-ACC',
                workspace_root / 'src' / 'qcar2_laneseg_acc',
            ]
            
            for src_path in src_patterns:
                if (src_path / 'package.xml').exists():
                    self.get_logger().info(f'Found source package at: {src_path}')
                    return src_path
        
        # Strategy 3: Search in current working directory
        cwd_path = Path.cwd()
        for _ in range(10):
            package_xml = cwd_path / 'package.xml'
            if package_xml.exists():
                self.get_logger().info(f'Found package.xml in cwd: {cwd_path}')
                return cwd_path
            
            parent = cwd_path.parent
            if parent == cwd_path:
                break
            cwd_path = parent
        
        return None
    
    def image_callback(self, msg):
        """
        Callback function for camera image messages.
        
        Args:
            msg (sensor_msgs.msg.Image): Incoming image message
        """
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # On first frame, detect and log resolution
            if self.first_frame:
                height, width = cv_image.shape[:2]
                self.get_logger().info('='*60)
                self.get_logger().info(f'IMAGE RESOLUTION DETECTED: {width}x{height}')
                self.get_logger().info(f'Width: {width} pixels | Height: {height} pixels')
                self.get_logger().info('IMPORTANT: Configure your U-Net input size accordingly!')
                self.get_logger().info('='*60)
                self.first_frame = False
            
            # Throttling mechanism - only save if enough time has passed
            current_time = time.time()
            time_since_last_capture = current_time - self.last_capture_time
            
            if time_since_last_capture >= self.throttle_interval:
                # Generate timestamp-based filename (milliseconds for uniqueness)
                timestamp_ms = int(current_time * 1000)
                filename = f'img_{timestamp_ms}.png'
                filepath = os.path.join(self.output_path, filename)
                
                # Save image
                cv2.imwrite(filepath, cv_image)
                
                # Update counters
                self.last_capture_time = current_time
                self.image_count += 1
                
                self.get_logger().info(
                    f'[{self.image_count}] Saved: {filename} '
                    f'(Δt={time_since_last_capture:.2f}s)'
                )
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            # Don't crash the node - continue processing next frames


def main(args=None):
    """Main function to initialize and run the node."""
    rclpy.init(args=args)
    
    try:
        node = ImageCaptureNode()
        
        # Print startup message
        print("\n" + "="*60)
        print("IMAGE CAPTURE NODE FOR U-NET TRAINING DATA ACQUISITION")
        print("="*60)
        print("Press Ctrl+C to stop capturing and exit")
        print("="*60 + "\n")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested - stopping capture...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'node' in locals():
            print(f"\nTotal images captured: {node.image_count}")
            print(f"Images saved to: {node.output_path}\n")
            node.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
