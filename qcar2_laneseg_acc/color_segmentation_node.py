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
from scipy.ndimage import gaussian_filter

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
        self.declare_parameter('roi_height_ratio', 0.35)
        self.declare_parameter('num_samples', 10)
        self.declare_parameter('sample_region_size', 4)  # Increased for better sampling
        self.declare_parameter('lut_filename', 'color_lut.npy')
        self.declare_parameter('input_image_topic', '/camera/color_image')
        self.declare_parameter('output_mask_topic', '/segmentation/color_mask')
        self.declare_parameter('clahe_clip_limit', 1.05)
        self.declare_parameter('clahe_tile_size', 300)
        self.declare_parameter('edge_kernel_size', 3)
        self.declare_parameter('enable_edge_detection', True)  # New: toggle edge detection
        self.declare_parameter('debug_logging', True)  # New: enable debug logs
        self.declare_parameter('smoothing_sigma', 5.0)  # Gaussian smoothing sigma for LUT generalization
        
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
        self.enable_edge_detection = self.get_parameter('enable_edge_detection').value
        self.debug_logging = self.get_parameter('debug_logging').value
        self.smoothing_sigma = self.get_parameter('smoothing_sigma').value
        
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
        self.calibration_frame_hsv = None  # Stores HSV+CLAHE normalized image
        self.calibration_frame_display = None  # BGR for display
        self.mouse_pos = (0, 0)
        
        # Debug: frame counter for periodic logging
        self.frame_count = 0
        
        # Find package paths
        self.pkg_share_dir = get_package_share_directory('qcar2_laneseg_acc')
        self.lut_path = os.path.join(self.pkg_share_dir, 'config', self.lut_filename)
        
        # Try to load existing LUT
        if self._load_lut():
            self.get_logger().info(f"LUT loaded from: {self.lut_path}")
            self._debug_analyze_lut()  # Analyze LUT on load
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
        self.get_logger().info(f"  Debug logging: {self.debug_logging}")
        self.get_logger().info(f"  Edge detection: {self.enable_edge_detection}")
    
    # =====================================================================
    # Debug Functions
    # =====================================================================
    def _debug_analyze_lut(self):
        """Analyze and log LUT statistics."""
        if self.lut is None:
            return
        
        self.get_logger().info("=== LUT Analysis ===")
        self.get_logger().info(f"  Shape: {self.lut.shape}")
        unique, counts = np.unique(self.lut, return_counts=True)
        total_bins = self.lut.size
        
        for label, count in zip(unique, counts):
            pct = 100.0 * count / total_bins
            class_name = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else f"unknown_{label}"
            self.get_logger().info(f"  Class {label} ({class_name}): {count} bins ({pct:.2f}%)")
    
    def _debug_log_hsv_stats(self, hsv_image, context=""):
        """Log HSV channel statistics."""
        if not self.debug_logging:
            return
        h, s, v = cv2.split(hsv_image)
        self.get_logger().info(
            f"[{context}] HSV stats: H=[{h.min()}-{h.max()}], "
            f"S=[{s.min()}-{s.max()}], V=[{v.min()}-{v.max()}]"
        )
    
    def _debug_log_mask_stats(self, mask):
        """Log mask label statistics."""
        if not self.debug_logging:
            return
        unique, counts = np.unique(mask, return_counts=True)
        stats = ", ".join([f"{label}:{count}" for label, count in zip(unique, counts)])
        self.get_logger().info(f"[Mask] Labels: {stats}")
    
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
        """
        Generate LUT from calibration histograms with Gaussian smoothing.
        
        The Gaussian smoothing expands the influence of sampled points to their
        neighbors in HSV space, making the segmentation more robust to small
        variations in illumination and color.
        """
        self.get_logger().info("Generating LUT from calibration data...")
        self.get_logger().info(f"  Using Gaussian smoothing with sigma={self.smoothing_sigma}")
        
        # LUT dimensions: H(180) x S(256) x V(256)
        # Each entry contains the class label (0-2)
        lut_shape = (180, 256, 256)
        
        # Accumulate histograms per class
        histograms = []
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_hist = np.zeros(lut_shape, dtype=np.float32)
            total_pixels = 0
            
            for sample in self.calibration_data[class_name]:
                # sample is HSV pixels array (N, 3)
                for h, s, v in sample:
                    class_hist[h, s, v] += 1
                    total_pixels += 1
            
            self.get_logger().info(f"  Class {class_idx} ({class_name}): {total_pixels} total pixels sampled")
            
            # Log HSV ranges for this class
            if total_pixels > 0:
                all_pixels = np.vstack(self.calibration_data[class_name])
                h_vals, s_vals, v_vals = all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2]
                self.get_logger().info(
                    f"    HSV ranges: H=[{h_vals.min()}-{h_vals.max()}], "
                    f"S=[{s_vals.min()}-{s_vals.max()}], V=[{v_vals.min()}-{v_vals.max()}]"
                )
            
            # ============================================================
            # GAUSSIAN SMOOTHING: Expand influence of samples to neighbors
            # ============================================================
            # Apply 3D Gaussian filter BEFORE normalization
            # This spreads the sampled points to nearby HSV values,
            # making the LUT more tolerant to illumination variations
            if self.smoothing_sigma > 0 and total_pixels > 0:
                class_hist = gaussian_filter(
                    class_hist, 
                    sigma=self.smoothing_sigma,
                    mode='wrap'  # Wrap H dimension (hue is circular)
                )
                self.get_logger().info(f"    Applied Gaussian smoothing (sigma={self.smoothing_sigma})")
            
            # Normalize histogram AFTER smoothing
            total = class_hist.sum()
            if total > 0:
                class_hist /= total
            
            histograms.append(class_hist)
        
        # Stack histograms and find max probability class for each bin
        histograms = np.stack(histograms, axis=-1)  # (180, 256, 256, 3)
        
        # Get max probability per bin
        max_probs = np.max(histograms, axis=-1)
        
        # Assign each bin to class with highest probability
        self.lut = np.argmax(histograms, axis=-1).astype(np.uint8)
        
        # Count bins that have actual probability (were influenced by samples)
        has_samples_mask = max_probs > 0
        num_covered = has_samples_mask.sum()
        coverage_pct = 100.0 * num_covered / self.lut.size
        self.get_logger().info(f"  LUT coverage after smoothing: {num_covered} bins ({coverage_pct:.2f}%)")
        
        # For bins with no samples (even after smoothing), default to sidewalk (0)
        no_samples_mask = max_probs == 0
        num_empty = no_samples_mask.sum()
        self.get_logger().info(f"  Empty bins (no influence): {num_empty} ({100.0*num_empty/self.lut.size:.2f}%)")
        self.lut[no_samples_mask] = 0  # Default to sidewalk for unknown areas
        
        # Transfer to GPU if available
        if CUPY_AVAILABLE:
            self.lut_gpu = cp.asarray(self.lut)
        
        self.get_logger().info("LUT generation complete!")
        self._debug_analyze_lut()
    
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
        sidewalk_mask = (mask == 0).astype(np.uint8) * 255
        
        # Only detect edges where road meets sidewalk
        # Use smaller kernel for more precise edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,  # Changed from ELLIPSE for less noise
            (3, 3)  # Smaller kernel
        )
        
        # Dilate road slightly
        road_dilated = cv2.dilate(road_mask, kernel, iterations=1)
        
        # Find where dilated road overlaps with sidewalk
        road_edge = cv2.bitwise_and(road_dilated, sidewalk_mask)
        
        # Clean up noise with morphological opening
        road_edge = cv2.morphologyEx(road_edge, cv2.MORPH_OPEN, kernel)
        
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
            if self.calibration_frame_hsv is not None:
                self._collect_sample(x, y)
    
    def _collect_sample(self, x, y):
        """Collect HSV sample at clicked position from CLAHE-normalized HSV image."""
        half_size = self.sample_region_size // 2
        
        # Get region bounds from the NORMALIZED HSV image
        h, w = self.calibration_frame_hsv.shape[:2]
        x1 = max(0, x - half_size)
        x2 = min(w, x + half_size)
        y1 = max(0, y - half_size)
        y2 = min(h, y + half_size)
        
        # Extract region directly from HSV+CLAHE normalized image
        # This ensures calibration uses the SAME values as inference
        hsv_region = self.calibration_frame_hsv[y1:y2, x1:x2]
        
        # Flatten to list of HSV pixels
        hsv_pixels = hsv_region.reshape(-1, 3)
        
        # Store sample
        current_class = self.CLASS_NAMES[self.current_class_idx]
        self.calibration_data[current_class].append(hsv_pixels)
        self.samples_collected += 1
        
        # Debug: log HSV range of this sample
        h_vals, s_vals, v_vals = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
        self.get_logger().info(
            f"Sample {self.samples_collected}/{self.num_samples} for '{current_class}': "
            f"{len(hsv_pixels)} pixels, H=[{h_vals.min()}-{h_vals.max()}], "
            f"S=[{s_vals.min()}-{s_vals.max()}], V=[{v_vals.min()}-{v_vals.max()}]"
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
    
    def _run_calibration_gui(self, roi_bgr, hsv_normalized):
        """Display calibration GUI with CLAHE-normalized HSV stored for sampling."""
        # Store the normalized HSV for sampling (same as inference pipeline)
        self.calibration_frame_hsv = hsv_normalized.copy()
        self.calibration_frame_display = roi_bgr.copy()
        display = roi_bgr.copy()
        
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
        # Convert BGR to more visible color for text
        text_color = (255, 255, 255) if current_class == 'sidewalk' else color
        
        text = f"Click on {current_class.upper()} ({self.samples_collected}/{self.num_samples})"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(display, "Press 'q' to cancel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Sample size: {self.sample_region_size}x{self.sample_region_size}px", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
            
            # Apply preprocessing BEFORE calibration (same as inference)
            # 1. Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 2. Apply CLAHE normalization
            hsv_normalized = self._apply_clahe(hsv)
            
            if self.calibration_mode:
                # Run calibration GUI with preprocessed image
                self._run_calibration_gui(roi, hsv_normalized)
                return
            
            # Normal segmentation pipeline
            # HSV + CLAHE already applied above
            
            # Debug: log HSV stats periodically (every 30 frames)
            self.frame_count += 1
            if self.debug_logging and self.frame_count % 30 == 0:
                self._debug_log_hsv_stats(hsv_normalized, "Inference")
            
            # 3. Apply LUT segmentation
            mask = self._segment_with_lut(hsv_normalized)
            
            # Debug: log mask stats periodically
            if self.debug_logging and self.frame_count % 30 == 0:
                self._debug_log_mask_stats(mask)
            
            # 4. Detect road edge (optional)
            if self.enable_edge_detection:
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
