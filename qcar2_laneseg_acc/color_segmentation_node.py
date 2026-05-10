#!/usr/bin/env python3
"""
Right-Side Edge Detection Node

Real-time detection of pseudo-straight edges on the right side of the
image. No prior calibration or .npy LUT is required.  All filter stages
are fully parametrized via ROS2 dynamic reconfiguration
(rqt_reconfigure / ros2 param set).

Pipeline:
     1. Fisheye Undistortion      — Correct lens distortion            (optional)
     2. Adaptive Gamma (AGCWD)    — Normalize illumination             (optional, 18 params)
     3. ROI Crop                  — Keep bottom portion of image
     4. Gaussian Pre-blur         — Reduce noise before edge detect    (optional)
     5. Grayscale + CLAHE         — Equalize contrast
     6. Canny Edge Detection      — Extract edges
     7. Morphological Cleanup     — Close + Open to clean edges        (optional)
     8. Hough Line Detection      — Find pseudo-straight lines
     9. Right-Side Filtering      — Keep only lines on right half
    10. Robot Mask                — Zero-fill robot-visible region      (optional)
    11. Publish                   — CompressedImage (JPEG)

Output: Binary mask with detected right-side edges drawn in Red.

Author: hackbrian (+ improvements Eduardex)
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
# scipy.ndimage.gaussian_filter ya no se necesita (era para LUT)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[WARN] CuPy not available, falling back to NumPy (CPU)")


def _odd_ksize(k: int) -> int:
    """Ensure kernel size is a positive odd integer."""
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


class ColorSegmentationNode(Node):
    """
    ROS2 Node — Black/White border edge detection.

    Detects edges that separate a dark region from a bright region
    (e.g. black court floor vs white wall) using Canny as candidate
    generator and per-pixel gradient-direction intensity validation.
    No prior calibration or .npy file needed.
    """

    # Color for validated edge pixels on the output mask
    EDGE_COLOR = (0, 0, 255)  # Red in BGR

    # =================================================================
    # Initialization
    # =================================================================
    def __init__(self):
        super().__init__('color_segmentation_node')

        # --- Topics & I/O ---
        self.declare_parameter('input_image_topic', '/camera/color_image/compressed')
        self.declare_parameter('output_mask_topic', '/segmentation/color_mask/compressed')
        self.declare_parameter('canny_debug_topic', '/segmentation/canny_debug/compressed')
        self.declare_parameter('jpeg_quality', 10)
        self.declare_parameter('debug_logging', True)

        # --- ROI ---
        self.declare_parameter('roi_height_ratio', 0.2)

        # --- Pre-blur ---
        self.declare_parameter('enable_pre_blur', True)
        self.declare_parameter('pre_blur_ksize', 9)
        self.declare_parameter('pre_blur_sigma', 0.0)

        # --- CLAHE ---
        self.declare_parameter('clahe_clip_limit', 5.05)
        self.declare_parameter('clahe_tile_size', 10)

        # --- Canny Edge Detection ---
        self.declare_parameter('canny_threshold1', 50)
        self.declare_parameter('canny_threshold2', 150)
        self.declare_parameter('canny_aperture', 3)

        # --- Black/White Edge Validation ---
        self.declare_parameter('use_black_white_edge_filter', True)
        self.declare_parameter('dark_thresh', 80)
        self.declare_parameter('bright_thresh', 160)
        self.declare_parameter('contrast_thresh', 70)
        self.declare_parameter('sample_offset_px', 5)
        self.declare_parameter('min_edge_pixels', 50)

        # --- Orientation Filter (reject vertical lines via gradient angle) ---
        self.declare_parameter('enable_orientation_filter', True)
        self.declare_parameter('vertical_reject_deg', 25.0)

        # --- Connected-Component Filter (reject small / vertical blobs) ---
        self.declare_parameter('enable_component_filter', True)
        self.declare_parameter('min_component_area', 80)
        self.declare_parameter('min_component_width', 8)
        self.declare_parameter('max_vertical_angle_deg', 20.0)

        # --- Post-Validation Morphology (engrosamiento de bordes) ---
        self.declare_parameter('edge_close_size', 7)
        self.declare_parameter('edge_close_iter', 2)
        self.declare_parameter('edge_dilate_size', 5)
        self.declare_parameter('edge_dilate_iter', 2)

        # --- Morphological Cleanup ---
        self.declare_parameter('enable_morph_cleanup', True)
        self.declare_parameter('morph_kernel_size', 5)
        self.declare_parameter('morph_close_iter', 2)
        self.declare_parameter('morph_open_iter', 1)

        # --- Robot Mask ---
        self.declare_parameter('enable_robot_mask', False)
        self.declare_parameter('robot_mask_x1', 400)
        self.declare_parameter('robot_mask_y1', 430)
        self.declare_parameter('robot_mask_x2', 530)
        self.declare_parameter('robot_mask_y2', 480)

        # --- Adaptive Gamma Correction (AGCWD) ---
        self.declare_parameter('enable_adapt_gamma', True)
        self.declare_parameter('agcwd_base_alpha', 0.25)
        self.declare_parameter('agcwd_mean_v_low', 110.0)
        self.declare_parameter('agcwd_mean_v_high', 130.0)
        self.declare_parameter('agcwd_p95_v_thresh', 240.0)
        self.declare_parameter('agcwd_sat_ratio_thresh', 0.02)
        self.declare_parameter('agcwd_dimmed_alpha', 0.85)
        self.declare_parameter('agcwd_bright_alpha', 0.70)
        self.declare_parameter('agcwd_highlight_v_thresh', 240)
        self.declare_parameter('agcwd_highlight_s_thresh', 80)
        self.declare_parameter('agcwd_highlight_strength', 0.45)
        self.declare_parameter('agcwd_sat_morph_ksize', 5)
        self.declare_parameter('agcwd_sat_median_ksize', 5)
        self.declare_parameter('agcwd_dimmed_sat_alpha', 1.5)
        self.declare_parameter('agcwd_dimmed_sat_beta', 10.0)
        self.declare_parameter('agcwd_bright_sat_alpha', 2.0)
        self.declare_parameter('agcwd_bright_sat_beta', 15.0)
        self.declare_parameter('agcwd_neutral_sat_alpha', 1.4)
        self.declare_parameter('agcwd_neutral_sat_beta', 5.0)

        # --- Fisheye Undistortion ---
        self.declare_parameter('enable_fisheye_undistortion', True)
        self.declare_parameter('fisheye_calibration_file', '')
        self.declare_parameter('undistort_balance', 0.0)
        self.declare_parameter('undistort_scale', 1.0)
        self.declare_parameter('undistorted_compressed_topic',
                               '/qcar2/csi/undistorted/image/compressed')
        self.declare_parameter('publish_undistorted', True)
        self.declare_parameter('undistorted_jpeg_quality', 80)

        # =============================================================
        # Read parameters
        # =============================================================
        self.roi_height_ratio = float(self.get_parameter('roi_height_ratio').value)
        input_topic = str(self.get_parameter('input_image_topic').value)
        output_topic = str(self.get_parameter('output_mask_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_logging = bool(self.get_parameter('debug_logging').value)

        clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        tile_size = int(self.get_parameter('clahe_tile_size').value)

        self.enable_pre_blur = bool(self.get_parameter('enable_pre_blur').value)
        self.pre_blur_ksize = _odd_ksize(self.get_parameter('pre_blur_ksize').value)
        self.pre_blur_sigma = float(self.get_parameter('pre_blur_sigma').value)

        self.enable_robot_mask = bool(self.get_parameter('enable_robot_mask').value)
        self.robot_mask_x1 = int(self.get_parameter('robot_mask_x1').value)
        self.robot_mask_y1 = int(self.get_parameter('robot_mask_y1').value)
        self.robot_mask_x2 = int(self.get_parameter('robot_mask_x2').value)
        self.robot_mask_y2 = int(self.get_parameter('robot_mask_y2').value)

        self.enable_morph_cleanup = bool(self.get_parameter('enable_morph_cleanup').value)
        self.morph_kernel_size = _odd_ksize(self.get_parameter('morph_kernel_size').value)
        self.morph_close_iter = int(self.get_parameter('morph_close_iter').value)
        self.morph_open_iter = int(self.get_parameter('morph_open_iter').value)

        self.enable_adapt_gamma = bool(self.get_parameter('enable_adapt_gamma').value)

        # AGCWD params
        self.agcwd_base_alpha = float(self.get_parameter('agcwd_base_alpha').value)
        self.agcwd_mean_v_low = float(self.get_parameter('agcwd_mean_v_low').value)
        self.agcwd_mean_v_high = float(self.get_parameter('agcwd_mean_v_high').value)
        self.agcwd_p95_v_thresh = float(self.get_parameter('agcwd_p95_v_thresh').value)
        self.agcwd_sat_ratio_thresh = float(self.get_parameter('agcwd_sat_ratio_thresh').value)
        self.agcwd_dimmed_alpha = float(self.get_parameter('agcwd_dimmed_alpha').value)
        self.agcwd_bright_alpha = float(self.get_parameter('agcwd_bright_alpha').value)
        self.agcwd_highlight_v_thresh = int(self.get_parameter('agcwd_highlight_v_thresh').value)
        self.agcwd_highlight_s_thresh = int(self.get_parameter('agcwd_highlight_s_thresh').value)
        self.agcwd_highlight_strength = float(self.get_parameter('agcwd_highlight_strength').value)
        self.agcwd_sat_morph_ksize = _odd_ksize(self.get_parameter('agcwd_sat_morph_ksize').value)
        self.agcwd_sat_median_ksize = _odd_ksize(self.get_parameter('agcwd_sat_median_ksize').value)
        self.agcwd_dimmed_sat_alpha = float(self.get_parameter('agcwd_dimmed_sat_alpha').value)
        self.agcwd_dimmed_sat_beta = float(self.get_parameter('agcwd_dimmed_sat_beta').value)
        self.agcwd_bright_sat_alpha = float(self.get_parameter('agcwd_bright_sat_alpha').value)
        self.agcwd_bright_sat_beta = float(self.get_parameter('agcwd_bright_sat_beta').value)
        self.agcwd_neutral_sat_alpha = float(self.get_parameter('agcwd_neutral_sat_alpha').value)
        self.agcwd_neutral_sat_beta = float(self.get_parameter('agcwd_neutral_sat_beta').value)

        # Canny params
        self.canny_threshold1 = int(self.get_parameter('canny_threshold1').value)
        self.canny_threshold2 = int(self.get_parameter('canny_threshold2').value)
        self.canny_aperture = _odd_ksize(self.get_parameter('canny_aperture').value)

        # Black/White edge validation params
        self.use_bw_filter = bool(self.get_parameter('use_black_white_edge_filter').value)
        self.dark_thresh = int(self.get_parameter('dark_thresh').value)
        self.bright_thresh = int(self.get_parameter('bright_thresh').value)
        self.contrast_thresh = int(self.get_parameter('contrast_thresh').value)
        self.sample_offset_px = int(self.get_parameter('sample_offset_px').value)
        self.min_edge_pixels = int(self.get_parameter('min_edge_pixels').value)

        # Orientation filter params
        self.enable_orientation_filter = bool(
            self.get_parameter('enable_orientation_filter').value)
        self.vert_reject_deg = float(
            self.get_parameter('vertical_reject_deg').value)

        # Component filter params
        self.enable_component_filter = bool(
            self.get_parameter('enable_component_filter').value)
        self.min_component_area = int(
            self.get_parameter('min_component_area').value)
        self.min_component_width = int(
            self.get_parameter('min_component_width').value)
        self.max_vertical_angle_deg = float(
            self.get_parameter('max_vertical_angle_deg').value)

        # Post-validation morphology params
        self.edge_close_size = int(self.get_parameter('edge_close_size').value)
        self.edge_close_iter = int(self.get_parameter('edge_close_iter').value)
        self.edge_dilate_size = int(self.get_parameter('edge_dilate_size').value)
        self.edge_dilate_iter = int(self.get_parameter('edge_dilate_iter').value)

        # Fisheye params
        self._fisheye_enabled = bool(self.get_parameter('enable_fisheye_undistortion').value)
        self._fisheye_calib_file = str(self.get_parameter('fisheye_calibration_file').value)
        self._undistort_balance = float(self.get_parameter('undistort_balance').value)
        self._undistort_scale = float(self.get_parameter('undistort_scale').value)
        self._publish_undistorted = bool(self.get_parameter('publish_undistorted').value)
        self._undistorted_jpeg_quality = max(1, min(100, int(
            self.get_parameter('undistorted_jpeg_quality').value)))
        undistorted_topic = str(
            self.get_parameter('undistorted_compressed_topic').value)

        # Fisheye calibration state
        self._fisheye_K = None
        self._fisheye_D = None
        self._fisheye_map1 = None
        self._fisheye_map2 = None
        self._fisheye_map_size = None

        # =============================================================
        # Initialize components
        # =============================================================
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile_size, tile_size))
        self.frame_count = 0

        # =============================================================
        # Resolve config directory (solo para fisheye calibration)
        # =============================================================
        source_config = self._find_source_config_dir()
        if source_config:
            self._config_dir = source_config
        else:
            pkg_share = get_package_share_directory('qcar2_laneseg_acc')
            self._config_dir = os.path.join(pkg_share, 'config')

        if self._fisheye_enabled:
            self._load_fisheye_calibration()

        # =============================================================
        # QoS + Pub/Sub
        # =============================================================
        qos_in = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_out = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        canny_topic = str(self.get_parameter('canny_debug_topic').value)

        self.sub = self.create_subscription(
            CompressedImage, input_topic, self.image_callback, qos_in)
        self.pub = self.create_publisher(CompressedImage, output_topic, qos_out)
        self.canny_pub = self.create_publisher(CompressedImage, canny_topic, qos_out)
        self.undistorted_pub = self.create_publisher(
            CompressedImage, undistorted_topic, qos_out)

        # Startup log
        self.get_logger().info("Right-Side Edge Detection Node started")
        self.get_logger().info(f"  Input: {input_topic}")
        self.get_logger().info(f"  Output: {output_topic}")
        self.get_logger().info(f"  Pre-blur: {self.enable_pre_blur} (k={self.pre_blur_ksize})")
        self.get_logger().info(f"  Morph cleanup: {self.enable_morph_cleanup} (k={self.morph_kernel_size})")
        self.get_logger().info(f"  Adapt gamma: {self.enable_adapt_gamma}")
        self.get_logger().info(
            f"  Canny: t1={self.canny_threshold1} t2={self.canny_threshold2}")
        self.get_logger().info(
            f"  BW filter: dark<{self.dark_thresh} bright>{self.bright_thresh} "
            f"contrast>{self.contrast_thresh} offset={self.sample_offset_px}px")
        self.get_logger().info(f"  Canny debug: {canny_topic}")
        self.get_logger().info(
            f"  Fisheye: {self._fisheye_enabled} "
            f"(bal={self._undistort_balance}, scl={self._undistort_scale})")
        self.add_on_set_parameters_callback(self._parameter_callback)
    # =================================================================
    # Dynamic Parameter Callback
    # =================================================================
    def _parameter_callback(self, params):
        for param in params:
            n, v = param.name, param.value
            # ROI / General
            if n == 'roi_height_ratio':       self.roi_height_ratio = float(v)
            elif n == 'debug_logging':        self.debug_logging = bool(v)
            elif n == 'jpeg_quality':         self.jpeg_quality = max(1, min(100, int(v)))
            # CLAHE
            elif n == 'clahe_clip_limit':
                self.clahe = cv2.createCLAHE(
                    clipLimit=float(v),
                    tileGridSize=(int(self.get_parameter('clahe_tile_size').value),) * 2)
            elif n == 'clahe_tile_size':
                self.clahe = cv2.createCLAHE(
                    clipLimit=float(self.get_parameter('clahe_clip_limit').value),
                    tileGridSize=(int(v), int(v)))
            # Pre-blur
            elif n == 'enable_pre_blur':   self.enable_pre_blur = bool(v)
            elif n == 'pre_blur_ksize':    self.pre_blur_ksize = _odd_ksize(v)
            elif n == 'pre_blur_sigma':    self.pre_blur_sigma = float(v)
            # Robot mask
            elif n == 'enable_robot_mask': self.enable_robot_mask = bool(v)
            elif n == 'robot_mask_x1':     self.robot_mask_x1 = int(v)
            elif n == 'robot_mask_y1':     self.robot_mask_y1 = int(v)
            elif n == 'robot_mask_x2':     self.robot_mask_x2 = int(v)
            elif n == 'robot_mask_y2':     self.robot_mask_y2 = int(v)
            # Morph
            elif n == 'enable_morph_cleanup': self.enable_morph_cleanup = bool(v)
            elif n == 'morph_kernel_size':    self.morph_kernel_size = _odd_ksize(v)
            elif n == 'morph_close_iter':     self.morph_close_iter = int(v)
            elif n == 'morph_open_iter':      self.morph_open_iter = int(v)
            # Canny
            elif n == 'canny_threshold1':     self.canny_threshold1 = int(v)
            elif n == 'canny_threshold2':     self.canny_threshold2 = int(v)
            elif n == 'canny_aperture':       self.canny_aperture = _odd_ksize(v)
            # Black/White edge validation
            elif n == 'use_black_white_edge_filter': self.use_bw_filter = bool(v)
            elif n == 'dark_thresh':            self.dark_thresh = int(v)
            elif n == 'bright_thresh':          self.bright_thresh = int(v)
            elif n == 'contrast_thresh':        self.contrast_thresh = int(v)
            elif n == 'sample_offset_px':       self.sample_offset_px = int(v)
            elif n == 'min_edge_pixels':        self.min_edge_pixels = int(v)
            # Orientation filter
            elif n == 'enable_orientation_filter':  self.enable_orientation_filter = bool(v)
            elif n == 'vertical_reject_deg':        self.vert_reject_deg = float(v)
            # Component filter
            elif n == 'enable_component_filter':    self.enable_component_filter = bool(v)
            elif n == 'min_component_area':         self.min_component_area = int(v)
            elif n == 'min_component_width':        self.min_component_width = int(v)
            elif n == 'max_vertical_angle_deg':     self.max_vertical_angle_deg = float(v)
            # Post-validation morphology
            elif n == 'edge_close_size':        self.edge_close_size = int(v)
            elif n == 'edge_close_iter':        self.edge_close_iter = int(v)
            elif n == 'edge_dilate_size':       self.edge_dilate_size = int(v)
            elif n == 'edge_dilate_iter':       self.edge_dilate_iter = int(v)
            # AGCWD
            elif n == 'enable_adapt_gamma':       self.enable_adapt_gamma = bool(v)
            elif n == 'agcwd_base_alpha':         self.agcwd_base_alpha = float(v)
            elif n == 'agcwd_mean_v_low':         self.agcwd_mean_v_low = float(v)
            elif n == 'agcwd_mean_v_high':        self.agcwd_mean_v_high = float(v)
            elif n == 'agcwd_p95_v_thresh':       self.agcwd_p95_v_thresh = float(v)
            elif n == 'agcwd_sat_ratio_thresh':   self.agcwd_sat_ratio_thresh = float(v)
            elif n == 'agcwd_dimmed_alpha':       self.agcwd_dimmed_alpha = float(v)
            elif n == 'agcwd_bright_alpha':       self.agcwd_bright_alpha = float(v)
            elif n == 'agcwd_highlight_v_thresh': self.agcwd_highlight_v_thresh = int(v)
            elif n == 'agcwd_highlight_s_thresh': self.agcwd_highlight_s_thresh = int(v)
            elif n == 'agcwd_highlight_strength': self.agcwd_highlight_strength = float(v)
            elif n == 'agcwd_sat_morph_ksize':    self.agcwd_sat_morph_ksize = _odd_ksize(v)
            elif n == 'agcwd_sat_median_ksize':   self.agcwd_sat_median_ksize = _odd_ksize(v)
            elif n == 'agcwd_dimmed_sat_alpha':   self.agcwd_dimmed_sat_alpha = float(v)
            elif n == 'agcwd_dimmed_sat_beta':    self.agcwd_dimmed_sat_beta = float(v)
            elif n == 'agcwd_bright_sat_alpha':   self.agcwd_bright_sat_alpha = float(v)
            elif n == 'agcwd_bright_sat_beta':    self.agcwd_bright_sat_beta = float(v)
            elif n == 'agcwd_neutral_sat_alpha':  self.agcwd_neutral_sat_alpha = float(v)
            elif n == 'agcwd_neutral_sat_beta':   self.agcwd_neutral_sat_beta = float(v)
            # Fisheye
            elif n == 'enable_fisheye_undistortion':
                self._fisheye_enabled = bool(v)
                if self._fisheye_enabled and self._fisheye_K is None:
                    self._load_fisheye_calibration()
            elif n == 'undistort_balance':
                self._undistort_balance = float(v)
                self._fisheye_map1 = self._fisheye_map2 = self._fisheye_map_size = None
            elif n == 'undistort_scale':
                self._undistort_scale = float(v)
                self._fisheye_map1 = self._fisheye_map2 = self._fisheye_map_size = None
            elif n == 'publish_undistorted':
                self._publish_undistorted = bool(v)
            elif n == 'undistorted_jpeg_quality':
                self._undistorted_jpeg_quality = max(1, min(100, int(v)))

            self.get_logger().info(f"Parameter '{n}' updated to: {v}")
        return SetParametersResult(successful=True)

    # =================================================================
    # Source Directory Auto-Detection
    # =================================================================
    def _find_source_config_dir(self):
        try:
            share_dir = get_package_share_directory('qcar2_laneseg_acc')
            ws_root = share_dir
            for _ in range(4):
                ws_root = os.path.dirname(ws_root)
            src_dir = os.path.join(ws_root, 'src')
            if not os.path.isdir(src_dir):
                return None
            for root, dirs, files in os.walk(src_dir):
                depth = root[len(src_dir):].count(os.sep)
                if depth >= 2:
                    dirs.clear()
                    continue
                if 'package.xml' in files:
                    pkg_xml_path = os.path.join(root, 'package.xml')
                    with open(pkg_xml_path, 'r') as f:
                        if '<name>qcar2_laneseg_acc</name>' in f.read():
                            config_dir = os.path.join(root, 'config')
                            os.makedirs(config_dir, exist_ok=True)
                            return config_dir
        except Exception as e:
            self.get_logger().warn(f"Source config auto-detection failed: {e}")
        return None

    # =================================================================
    # Adaptive Gamma Correction (AGCWD) — instance methods
    # =================================================================
    def _image_agcwd(self, img_u8: np.ndarray, a: float = None,
                     truncated_cdf: bool = False) -> np.ndarray:
        """AGCWD on a single-channel uint8 image."""
        if a is None:
            a = self.agcwd_base_alpha
        if img_u8.dtype != np.uint8:
            img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

        hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).flatten().astype(np.float64)
        total = hist.sum()
        if total <= 0:
            return img_u8.copy()

        prob = hist / total
        prob_min, prob_max = prob.min(), prob.max()
        if abs(prob_max - prob_min) < 1e-12:
            return img_u8.copy()

        pn_temp = (prob - prob_min) / (prob_max - prob_min)
        pos_mask = pn_temp >= 0
        neg_mask = ~pos_mask

        pn_wd = np.zeros_like(pn_temp)
        pn_wd[pos_mask] = prob_max * np.power(pn_temp[pos_mask], a)
        pn_wd[neg_mask] = prob_max * (-np.power(-pn_temp[neg_mask], a))

        s = pn_wd.sum()
        if abs(s) < 1e-12:
            return img_u8.copy()

        prob_wd = pn_wd / s
        cdf_wd = np.cumsum(prob_wd)
        inverse_cdf = np.maximum(0.5, 1.0 - cdf_wd) if truncated_cdf else (1.0 - cdf_wd)

        lut = np.arange(256, dtype=np.float64)
        lut = np.round(255.0 * np.power(lut / 255.0, inverse_cdf))
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return cv2.LUT(img_u8, lut)

    def _process_bright(self, img_u8: np.ndarray) -> np.ndarray:
        """For very bright images."""
        img_negative = 255 - img_u8
        agcwd_neg = self._image_agcwd(img_negative, a=self.agcwd_bright_alpha,
                                       truncated_cdf=False)
        return 255 - agcwd_neg

    def _process_dimmed(self, img_u8: np.ndarray) -> np.ndarray:
        """For dark images."""
        return self._image_agcwd(img_u8, a=self.agcwd_dimmed_alpha,
                                  truncated_cdf=True)

    def _compress_highlights(self, v_channel: np.ndarray,
                             sat_mask: np.ndarray) -> np.ndarray:
        """Compress locally overexposed zones."""
        strength = self.agcwd_highlight_strength
        v = v_channel.astype(np.float32)
        out = v.copy()
        norm = v / 255.0
        out[sat_mask > 0] = 255.0 * np.power(norm[sat_mask > 0],
                                               1.0 / max(strength, 1e-3))
        return np.clip(out, 0, 255).astype(np.uint8)

    def _correct_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Robust illumination correction using parametrized AGCWD."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_v = float(np.mean(v))
        p95_v = float(np.percentile(v, 95))
        sat_ratio = float(np.mean(v > self.agcwd_highlight_v_thresh))

        # Overexposure mask
        sat_mask = ((v > self.agcwd_highlight_v_thresh) &
                    (s < self.agcwd_highlight_s_thresh)).astype(np.uint8) * 255
        k = _odd_ksize(self.agcwd_sat_morph_ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
        mk = _odd_ksize(self.agcwd_sat_median_ksize)
        sat_mask = cv2.medianBlur(sat_mask, mk)

        if mean_v < self.agcwd_mean_v_low:
            v_corr = self._process_dimmed(v)
            s_corr = cv2.convertScaleAbs(s, alpha=self.agcwd_dimmed_sat_alpha,
                                         beta=self.agcwd_dimmed_sat_beta)
        elif (mean_v > self.agcwd_mean_v_high or
              p95_v > self.agcwd_p95_v_thresh or
              sat_ratio > self.agcwd_sat_ratio_thresh):
            v_corr = self._process_bright(v)
            v_corr = self._compress_highlights(v_corr, sat_mask)
            s_corr = cv2.convertScaleAbs(s, alpha=self.agcwd_bright_sat_alpha,
                                         beta=self.agcwd_bright_sat_beta)
        else:
            v_corr = v.copy()
            s_corr = cv2.convertScaleAbs(s, alpha=self.agcwd_neutral_sat_alpha,
                                         beta=self.agcwd_neutral_sat_beta)

        hsv_corr = cv2.merge([h, s_corr, v_corr])
        return cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)
    # =================================================================
    # Fisheye Undistortion
    # =================================================================
    def _load_fisheye_calibration(self):
        calib_path = self._fisheye_calib_file
        if not calib_path:
            calib_path = os.path.join(self._config_dir, 'fisheye_calibration.yaml')
        if not os.path.isfile(calib_path):
            self.get_logger().warn(
                f"Fisheye calibration file not found: {calib_path} — "
                f"undistortion disabled.")
            self._fisheye_enabled = False
            return
        try:
            with open(calib_path, 'r') as f:
                calib = yaml.safe_load(f)
            km = calib.get('camera_matrix', {})
            k_data = km.get('data')
            if k_data is None or len(k_data) != 9:
                raise ValueError("camera_matrix.data must have 9 elements")
            self._fisheye_K = np.array(k_data, dtype=np.float64).reshape(3, 3)
            dc = calib.get('distortion_coefficients', {})
            d_data = dc.get('data')
            if d_data is None or len(d_data) != 4:
                raise ValueError("distortion_coefficients.data must have 4 elements")
            self._fisheye_D = np.array(d_data, dtype=np.float64).reshape(4, 1)
            self.get_logger().info(f"Fisheye calibration loaded from: {calib_path}")
            self.get_logger().info(f"  K =\n{self._fisheye_K}")
            self.get_logger().info(f"  D = {self._fisheye_D.flatten()}")
            self._fisheye_map1 = self._fisheye_map2 = self._fisheye_map_size = None
        except Exception as e:
            self.get_logger().warn(f"Failed to parse fisheye calibration: {e}")
            self._fisheye_enabled = False
            self._fisheye_K = self._fisheye_D = None

    def _compute_undistort_maps(self, w: int, h: int):
        K, D = self._fisheye_K, self._fisheye_D
        R = np.eye(3, dtype=np.float64)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), R, balance=self._undistort_balance)
        if abs(self._undistort_scale - 1.0) > 1e-6:
            new_K[0, 0] *= self._undistort_scale
            new_K[1, 1] *= self._undistort_scale
        self._fisheye_map1, self._fisheye_map2 = \
            cv2.fisheye.initUndistortRectifyMap(K, D, R, new_K, (w, h), cv2.CV_16SC2)
        self._fisheye_map_size = (w, h)
        self.get_logger().info(
            f"Fisheye maps computed for {w}x{h} "
            f"(balance={self._undistort_balance}, scale={self._undistort_scale})")

    def _undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self._fisheye_map1 is None or self._fisheye_map_size != (w, h):
            self._compute_undistort_maps(w, h)
        return cv2.remap(frame, self._fisheye_map1, self._fisheye_map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def _publish_undistorted_image(self, frame: np.ndarray, header):
        success, buf = cv2.imencode(
            '.jpg', frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._undistorted_jpeg_quality])
        if success:
            comp_msg = CompressedImage()
            comp_msg.header = header
            comp_msg.format = 'jpeg'
            comp_msg.data = np.array(buf).tobytes()
            self.undistorted_pub.publish(comp_msg)

    # =================================================================
    # Image Processing Pipeline
    # =================================================================
    def _crop_roi(self, image):
        h, w = image.shape[:2]
        crop_start = int(h * (1 - self.roi_height_ratio))
        return image[crop_start:, :], crop_start

    def _apply_clahe_gray(self, gray_image):
        """Apply CLAHE directly on grayscale image."""
        return self.clahe.apply(gray_image)

    def _morph_cleanup_edges(self, edge_mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup on binary edge mask."""
        ks = _odd_ksize(self.morph_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        result = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel,
                                  iterations=self.morph_close_iter)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel,
                                  iterations=self.morph_open_iter)
        return result

    # -----------------------------------------------------------------
    # Black/White edge validation using gradient-direction sampling
    # -----------------------------------------------------------------
    def _validate_bw_edges(self, gray: np.ndarray,
                           edges: np.ndarray) -> np.ndarray:
        """
        For every Canny edge pixel, compute the Sobel gradient direction
        (normal to the edge), sample intensity on both sides at
        `sample_offset_px` distance, and accept the pixel only if one
        side is dark (< dark_thresh) and the other is bright
        (> bright_thresh) with |diff| > contrast_thresh.

        Returns a uint8 mask (255 = valid edge, 0 = rejected).
        """
        h, w = gray.shape
        offset = max(1, self.sample_offset_px)

        # Sobel gradients to get edge normal direction
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Coordinates of candidate edge pixels
        ey, ex = np.nonzero(edges)
        if ey.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # Gradient at edge pixels → normal direction
        gx_e = gx[ey, ex]
        gy_e = gy[ey, ex]
        mag = np.sqrt(gx_e ** 2 + gy_e ** 2) + 1e-6
        nx = (gx_e / mag)   # unit normal x
        ny = (gy_e / mag)   # unit normal y

        # Sample coordinates on side A (+normal) and side B (−normal)
        ax = np.clip(np.round(ex + nx * offset).astype(int), 0, w - 1)
        ay = np.clip(np.round(ey + ny * offset).astype(int), 0, h - 1)
        bx = np.clip(np.round(ex - nx * offset).astype(int), 0, w - 1)
        by = np.clip(np.round(ey - ny * offset).astype(int), 0, h - 1)

        # Intensities on each side
        ia = gray[ay, ax].astype(np.int16)
        ib = gray[by, bx].astype(np.int16)

        # Validation: one side dark AND the other bright AND enough contrast
        dark_a = ia < self.dark_thresh
        bright_a = ia > self.bright_thresh
        dark_b = ib < self.dark_thresh
        bright_b = ib > self.bright_thresh
        diff = np.abs(ia - ib)

        valid = ((dark_a & bright_b) | (dark_b & bright_a)) & \
                (diff > self.contrast_thresh)

        # Build validated mask
        result = np.zeros((h, w), dtype=np.uint8)
        result[ey[valid], ex[valid]] = 255
        return result

    # -----------------------------------------------------------------
    # Connected-component filter: reject small / vertical edge blobs
    # -----------------------------------------------------------------
    def _filter_components(self, mask: np.ndarray) -> np.ndarray:
        """
        Separate each edge into individual connected components using
        findContours.  For each component:
          - Reject if area < min_component_area
          - Reject if bounding-box width < min_component_width
          - Compute principal orientation with cv2.fitLine (PCA-based)
          - Reject if the fitted line is nearly vertical:
            |angle_from_horizontal| > (90 - max_vertical_angle_deg)
            i.e. the line deviates less than max_vertical_angle_deg
            from the Y axis.
        Rebuild mask with only accepted components.
        """
        h, w = mask.shape
        output = np.zeros_like(mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_component_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.min_component_width:
                continue

            # Fit a line through the contour points (PCA direction)
            # fitLine returns (vx, vy, x0, y0)
            if len(cnt) < 5:
                continue
            [vx, vy, _, _] = cv2.fitLine(
                cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)

            # Angle of the fitted line w.r.t. horizontal (0°=horiz, 90°=vert)
            angle_deg = abs(np.degrees(np.arctan2(abs(vy), abs(vx))))

            # Reject if the component line is too vertical
            if angle_deg > (90.0 - self.max_vertical_angle_deg):
                continue

            # Component accepted — draw it on the output
            cv2.drawContours(output, [cnt], -1, 255, thickness=cv2.FILLED)

        return output

    def _detect_bw_edges(self, roi_bgr: np.ndarray,
                         header) -> np.ndarray:
        """
        Full edge-detection pipeline with black/white validation:
          1. Grayscale + CLAHE
          2. Canny  (candidate edges)
          3. Publish raw Canny on debug topic
          4. Morph cleanup on Canny (optional)
          5. Per-pixel gradient-direction intensity validation
          6. Optional dilation for visibility
          7. Paint validated edges in red on a black BGR mask
        """
        h, w = roi_bgr.shape[:2]

        # 1) Grayscale + CLAHE
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = self._apply_clahe_gray(gray)

        # 2) Canny edge detection (candidate generator)
        aperture = _odd_ksize(self.canny_aperture)
        if aperture < 3:
            aperture = 3
        edges = cv2.Canny(gray_eq, self.canny_threshold1,
                          self.canny_threshold2, apertureSize=aperture)

        # 3) Publish raw Canny for calibration
        self._publish_canny_debug(edges, header)

        # 4) Morphological cleanup on Canny edges (optional)
        if self.enable_morph_cleanup:
            edges = self._morph_cleanup_edges(edges)

        # 5) Black/White intensity validation
        #    Uses original gray (NOT equalized) for accurate intensity
        if self.use_bw_filter:
            validated = self._validate_bw_edges(gray, edges)
        else:
            validated = edges  # Bypass: keep all Canny edges

        # 5b) Orientation filter — reject vertical LINES
        #     atan2(gy, gx) gives the GRADIENT direction (0-180°).
        #     Vertical lines have horizontal gradients: angle near 0° or 180°.
        #     Reject if angle <= vertical_reject_deg OR angle >= 180 - vertical_reject_deg.
        if self.enable_orientation_filter and np.count_nonzero(validated) > 0:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            angle_map = np.degrees(np.arctan2(gy, gx))  # -180..180
            angle_map = np.mod(angle_map, 180.0)         # 0..180

            vy, vx = np.nonzero(validated)
            angles = angle_map[vy, vx]
            # Horizontal gradient = vertical line → reject
            is_vertical_line = (angles <= self.vert_reject_deg) | \
                               (angles >= 180.0 - self.vert_reject_deg)
            validated[vy[is_vertical_line], vx[is_vertical_line]] = 0

        # 5c) Connected-component filter — reject small or vertical blobs
        #     Uses findContours + fitLine to compute the principal
        #     orientation of each edge fragment independently, BEFORE
        #     dilation/close so distinct edges are not merged.
        if self.enable_component_filter and np.count_nonzero(validated) > 0:
            validated = self._filter_components(validated)

        # 5d) Minimum pixel count — discard if too few validated pixels
        if np.count_nonzero(validated) < self.min_edge_pixels:
            h, w = roi_bgr.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        # 6) Post-validation morphology: close gaps + dilate to thicken
        #    CLOSE (dilate→erode) connects nearby fragments into
        #    continuous lines without adding much thickness.
        cs = _odd_ksize(self.edge_close_size)
        if cs >= 3 and self.edge_close_iter > 0:
            ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs))
            validated = cv2.morphologyEx(
                validated, cv2.MORPH_CLOSE, ck,
                iterations=self.edge_close_iter)

        #    DILATE thickens the surviving edge line for visibility.
        ds = _odd_ksize(self.edge_dilate_size)
        if ds >= 3 and self.edge_dilate_iter > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ds, ds))
            validated = cv2.dilate(
                validated, dk, iterations=self.edge_dilate_iter)

        # 7) Paint validated edge pixels in red
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[validated > 0] = self.EDGE_COLOR

        return output

    def _publish_canny_debug(self, edges: np.ndarray, header):
        """Publish raw Canny edges as a compressed grayscale image."""
        success, buf = cv2.imencode(
            '.jpg', edges,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if success:
            msg = CompressedImage()
            msg.header = header
            msg.format = 'jpeg'
            msg.data = np.array(buf).tobytes()
            self.canny_pub.publish(msg)

    def _apply_robot_mask_output(self, full_mask: np.ndarray) -> np.ndarray:
        h, w = full_mask.shape[:2]
        x1 = max(0, min(self.robot_mask_x1, w))
        x2 = max(0, min(self.robot_mask_x2, w))
        y1 = max(0, min(self.robot_mask_y1, h))
        y2 = max(0, min(self.robot_mask_y2, h))
        if x2 > x1 and y2 > y1:
            full_mask[y1:y2, x1:x2] = 0
        return full_mask

    # =================================================================
    # Main Callback
    # =================================================================
    def image_callback(self, msg):
        try:
            if not msg.data:
                return
            np_arr = np.frombuffer(msg.data, np.uint8)
            if np_arr.size == 0:
                return
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return

            # 0) Fisheye undistortion
            if self._fisheye_enabled and self._fisheye_K is not None:
                frame = self._undistort_frame(frame)
            if self._publish_undistorted:
                self._publish_undistorted_image(frame, msg.header)

            # 1) Adaptive Gamma Correction
            if self.enable_adapt_gamma:
                frame = self._correct_frame(frame)

            roi, crop_offset = self._crop_roi(frame)

            # 2) Pre-blur
            if self.enable_pre_blur and self.pre_blur_ksize > 1:
                roi = cv2.GaussianBlur(
                    roi, (self.pre_blur_ksize, self.pre_blur_ksize),
                    self.pre_blur_sigma)

            # 3) Edge detection con validación negro/blanco
            colored_mask = self._detect_bw_edges(roi, msg.header)

            self.frame_count += 1

            # 4) Compose full-frame output
            full_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            full_mask[crop_offset:, :] = colored_mask

            # 5) Robot mask
            if self.enable_robot_mask:
                full_mask = self._apply_robot_mask_output(full_mask)

            # 6) Publish
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, compressed_data = cv2.imencode('.jpg', full_mask, encode_params)
            out_msg = CompressedImage()
            out_msg.header = msg.header
            out_msg.format = 'jpeg'
            out_msg.data = compressed_data.tobytes()
            self.pub.publish(out_msg)

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
