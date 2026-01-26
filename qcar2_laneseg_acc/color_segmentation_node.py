#!/usr/bin/env python3
"""
HSV Color Segmentation Node with CuPy GPU Acceleration

This node performs real-time semantic segmentation using histogram-based 
color classification in HSV space with GPU acceleration via CuPy.

Pipeline:
1. ROI Crop - Keep bottom portion of image
2. HSV + CLAHE - Convert to HSV and normalize illumination
3. LUT Segmentation - Apply precomputed lookup table
4. Edge Detection - Find border between sidewalk and road
5. Publish segmentation mask

Author: hackbrian
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[WARN] CuPy not available, falling back to NumPy (CPU)")


class ColorSegmentationNode(Node):
    """
    ROS2 Node for HSV-based color segmentation with GUI calibration.
    
    Output Labels:
        0 - Sidewalk (Black mask)
        1 - Road (Blue mask)
        2 - Lane (Yellow mask)
        3 - Road Edge (Green mask)
    """
    
    # Mask colors in BGR format
    MASK_COLORS = {
        0: (0, 0, 0),       # Sidewalk - Black
        1: (255, 0, 0),     # Road - Blue
        2: (0, 255, 255),   # Lane - Yellow
        3: (0, 255, 0),     # Road Edge - Green
    }
    
    CLASS_NAMES = ['sidewalk', 'road', 'lane']
    
    def __init__(self):
        super().__init__('color_segmentation_node')
        
        # =================================================================
        # Parameters
        # =================================================================
        self.declare_parameter('roi_height_ratio', 0.9)
        self.declare_parameter('num_samples', 5)
        self.declare_parameter('sample_region_size', 3)
        self.declare_parameter('lut_filename', 'color_lut.npy')
        self.declare_parameter('input_image_topic', '/camera/color_image')
        self.declare_parameter('output_mask_topic', '/segmentation/color_mask')
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_tile_size', 8)
        self.declare_parameter('edge_kernel_size', 5)
        
        # Get parameters
        self.roi_height_ratio = self.get_parameter('roi_height_ratio').value
        self.num_samples = self.get_parameter('num_samples').value
        self.sample_region_size = self.get_parameter('sample_region_size').value
        self.lut_filename = self.get_parameter('lut_filename').value
        input_topic = self.get_parameter('input_image_topic').value
        output_topic = self.get_parameter('output_mask_topic').value
        clip_limit = self.get_parameter('clahe_clip_limit').value
        tile_size = self.get_parameter('clahe_tile_size').value
        self.edge_kernel_size = self.get_parameter('edge_kernel_size').value
        
        # =================================================================
        # Initialize components
        # =================================================================
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # LUT storage (will be loaded or generated)
        self.lut = None
        self.lut_gpu = None  # CuPy array for GPU lookup
        
        # Calibration state
        self.calibration_mode = False
        self.calibration_data = {cls: [] for cls in self.CLASS_NAMES}
        self.current_class_idx = 0
        self.samples_collected = 0
        self.calibration_frame = None
        self.mouse_pos = (0, 0)
        
        # Find package paths
        self.pkg_share_dir = get_package_share_directory('qcar2_laneseg_acc')
        self.lut_path = os.path.join(self.pkg_share_dir, 'config', self.lut_filename)
        
        # Try to load existing LUT
        if self._load_lut():
            self.get_logger().info(f"LUT loaded from: {self.lut_path}")
        else:
            self.get_logger().warn("No LUT found. Calibration mode will start on first image.")
            self.calibration_mode = True
        
        # =================================================================
        # QoS Profiles
        # =================================================================
        qos_input = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        qos_output = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # =================================================================
        # Subscriber and Publisher
        # =================================================================
        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            qos_input
        )
        
        self.pub = self.create_publisher(Image, output_topic, qos_output)
        
        self.get_logger().info(f"Color Segmentation Node started")
        self.get_logger().info(f"  Input: {input_topic}")
        self.get_logger().info(f"  Output: {output_topic}")
        self.get_logger().info(f"  CuPy GPU: {'Enabled' if CUPY_AVAILABLE else 'Disabled (CPU fallback)'}")
    
    # =====================================================================
    # LUT Management
    # =====================================================================
    def _load_lut(self) -> bool:
        """Load LUT from disk if exists."""
        if os.path.exists(self.lut_path):
            try:
                self.lut = np.load(self.lut_path)
                if CUPY_AVAILABLE:
                    self.lut_gpu = cp.asarray(self.lut)
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to load LUT: {e}")
        return False
    
    def _save_lut(self) -> bool:
        """Save LUT to disk."""
        try:
            # Ensure config directory exists
            config_dir = os.path.dirname(self.lut_path)
            os.makedirs(config_dir, exist_ok=True)
            
            np.save(self.lut_path, self.lut)
            self.get_logger().info(f"LUT saved to: {self.lut_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save LUT: {e}")
            return False
    
    def _generate_lut(self):
        """Generate LUT from calibration histograms."""
        self.get_logger().info("Generating LUT from calibration data...")
        
        # LUT dimensions: H(180) x S(256) x V(256)
        # Each entry contains the class label (0-2)
        lut_shape = (180, 256, 256)
        
        # Accumulate histograms per class
        histograms = []
        for class_name in self.CLASS_NAMES:
            class_hist = np.zeros(lut_shape, dtype=np.float32)
            
            for sample in self.calibration_data[class_name]:
                # sample is HSV pixels array (N, 3)
                for h, s, v in sample:
                    class_hist[h, s, v] += 1
            
            # Normalize histogram
            total = class_hist.sum()
            if total > 0:
                class_hist /= total
            
            histograms.append(class_hist)
        
        # Stack histograms and find max probability class for each bin
        histograms = np.stack(histograms, axis=-1)  # (180, 256, 256, 3)
        
        # Assign each bin to class with highest probability
        # Default to sidewalk (0) if no samples
        self.lut = np.argmax(histograms, axis=-1).astype(np.uint8)
        
        # Transfer to GPU if available
        if CUPY_AVAILABLE:
            self.lut_gpu = cp.asarray(self.lut)
        
        self.get_logger().info("LUT generation complete!")
    
    # =====================================================================
    # Image Processing Pipeline
    # =====================================================================
    def _crop_roi(self, image):
        """Crop image to keep bottom portion (ROI)."""
        h, w = image.shape[:2]
        crop_start = int(h * (1 - self.roi_height_ratio))
        return image[crop_start:, :], crop_start
    
    def _apply_clahe(self, hsv_image):
        """Apply CLAHE to V channel for illumination normalization."""
        h, s, v = cv2.split(hsv_image)
        v_equalized = self.clahe.apply(v)
        return cv2.merge([h, s, v_equalized])
    
    def _segment_with_lut(self, hsv_image):
        """Apply LUT segmentation using GPU if available."""
        h, s, v = cv2.split(hsv_image)
        
        if CUPY_AVAILABLE and self.lut_gpu is not None:
            # GPU path
            h_gpu = cp.asarray(h)
            s_gpu = cp.asarray(s)
            v_gpu = cp.asarray(v)
            
            # Lookup
            mask_gpu = self.lut_gpu[h_gpu, s_gpu, v_gpu]
            mask = cp.asnumpy(mask_gpu)
        else:
            # CPU path
            mask = self.lut[h, s, v]
        
        return mask
    
    def _detect_road_edge(self, mask):
        """Detect edge between sidewalk (0) and road (1)."""
        # Create binary mask: road vs non-road
        road_mask = (mask == 1).astype(np.uint8) * 255
        
        # Apply morphological gradient to find edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.edge_kernel_size, self.edge_kernel_size)
        )
        edge = cv2.morphologyEx(road_mask, cv2.MORPH_GRADIENT, kernel)
        
        # Only keep edges adjacent to sidewalk
        sidewalk_dilated = cv2.dilate(
            (mask == 0).astype(np.uint8) * 255, 
            kernel, 
            iterations=1
        )
        road_edge = cv2.bitwise_and(edge, sidewalk_dilated)
        
        return road_edge > 0
    
    def _colorize_mask(self, mask):
        """Convert label mask to colored visualization."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label, color in self.MASK_COLORS.items():
            colored[mask == label] = color
        
        return colored
    
    # =====================================================================
    # Calibration GUI
    # =====================================================================
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for calibration."""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.calibration_frame is not None:
                self._collect_sample(x, y)
    
    def _collect_sample(self, x, y):
        """Collect HSV sample at clicked position."""
        half_size = self.sample_region_size // 2
        
        # Get region bounds
        h, w = self.calibration_frame.shape[:2]
        x1 = max(0, x - half_size)
        x2 = min(w, x + half_size)
        y1 = max(0, y - half_size)
        y2 = min(h, y + half_size)
        
        # Extract region and convert to HSV
        region = self.calibration_frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Flatten to list of HSV pixels
        hsv_pixels = hsv_region.reshape(-1, 3)
        
        # Store sample
        current_class = self.CLASS_NAMES[self.current_class_idx]
        self.calibration_data[current_class].append(hsv_pixels)
        self.samples_collected += 1
        
        self.get_logger().info(
            f"Sample {self.samples_collected}/{self.num_samples} for '{current_class}' collected"
        )
        
        # Check if done with current class
        if self.samples_collected >= self.num_samples:
            self.samples_collected = 0
            self.current_class_idx += 1
            
            if self.current_class_idx >= len(self.CLASS_NAMES):
                # All classes calibrated
                self._finish_calibration()
    
    def _finish_calibration(self):
        """Finish calibration and generate LUT."""
        self.get_logger().info("Calibration complete! Generating LUT...")
        
        cv2.destroyAllWindows()
        
        self._generate_lut()
        self._save_lut()
        
        self.calibration_mode = False
        self.get_logger().info("Ready for segmentation!")
    
    def _run_calibration_gui(self, frame):
        """Display calibration GUI."""
        self.calibration_frame = frame.copy()
        display = frame.copy()
        
        # Draw sample region preview at mouse position
        half_size = self.sample_region_size // 2
        x, y = self.mouse_pos
        cv2.rectangle(
            display,
            (x - half_size, y - half_size),
            (x + half_size, y + half_size),
            (0, 255, 0), 2
        )
        
        # Draw instructions
        current_class = self.CLASS_NAMES[self.current_class_idx]
        color = self.MASK_COLORS[self.current_class_idx]
        
        text = f"Click on {current_class.upper()} ({self.samples_collected}/{self.num_samples})"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, "Press 'q' to cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Calibration", display)
        cv2.setMouseCallback("Calibration", self._mouse_callback)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().warn("Calibration cancelled")
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    # =====================================================================
    # Main Callback
    # =====================================================================
    def image_callback(self, msg):
        """Process incoming image."""
        try:
            # Convert ROS message to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if frame is None or frame.size == 0:
                return
            
            # Crop ROI
            roi, crop_offset = self._crop_roi(frame)
            
            if self.calibration_mode:
                # Run calibration GUI
                self._run_calibration_gui(roi)
                return
            
            # Normal segmentation pipeline
            # 1. Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 2. Apply CLAHE normalization
            hsv_normalized = self._apply_clahe(hsv)
            
            # 3. Apply LUT segmentation
            mask = self._segment_with_lut(hsv_normalized)
            
            # 4. Detect road edge
            road_edge = self._detect_road_edge(mask)
            mask[road_edge] = 3  # Road edge label
            
            # 5. Colorize mask
            colored_mask = self._colorize_mask(mask)
            
            # Reconstruct full image size (fill top with zeros)
            full_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            full_mask[crop_offset:, :] = colored_mask
            
            # Publish
            mask_msg = self.bridge.cv2_to_imgmsg(full_mask, encoding='bgr8')
            mask_msg.header = msg.header
            self.pub.publish(mask_msg)
            
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ColorSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
