#!/usr/bin/env python3
"""
HSV Color Segmentation Node with CuPy GPU Acceleration

Real-time semantic segmentation using histogram-based color classification
in HSV space. All filter stages are fully parametrized via ROS2 dynamic
reconfiguration (rqt_reconfigure / ros2 param set).

Pipeline:
     1. Fisheye Undistortion      — Correct lens distortion            (optional)
     2. Adaptive Gamma (AGCWD)    — Normalize illumination             (optional, 18 params)
     3. ROI Crop                  — Keep bottom portion of image
     4. Gaussian Pre-blur         — Reduce noise before HSV            (optional)
     5. HSV + CLAHE               — Convert to HSV & equalize V channel
     6. LUT Segmentation          — Lookup table with per-label
                                    probabilistic confidence filtering  (3 thresholds)
     7. Morphological Cleanup     — Close + Open to fill holes/noise   (optional)
     8. Edge Detection            — Sidewalk ↔ Road border             (optional)
     9. Colorize Mask             — Label → BGR color mapping
    10. Robot Mask                — Zero-fill robot-visible region      (optional)
    11. Publish                   — CompressedImage (JPEG)

Output Labels:
    0 - Sidewalk  (Black)
    1 - Road      (Blue)
    2 - Lane      (Yellow)
    3 - Road Edge (Red)

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
from scipy.ndimage import gaussian_filter
from sensor_msgs.msg import CompressedImage

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
    ROS2 Node for HSV-based color segmentation with GUI calibration.

    Output Labels:
        0 - Sidewalk (Black mask)
        1 - Road (Blue mask)
        2 - Lane (Yellow mask)
        3 - Road Edge (Red mask)
    """

    MASK_COLORS = {
        0: (0, 0, 0),       # Sidewalk - Black
        1: (255, 0, 0),     # Road - Blue
        2: (0, 255, 255),   # Lane - Yellow
        3: (0, 0, 255),     # Road Edge - Red
    }

    CLASS_NAMES = ['sidewalk', 'road', 'lane']

    # =================================================================
    # Initialization
    # =================================================================
    def __init__(self):
        super().__init__('color_segmentation_node')

        # --- Topics & I/O ---
        self.declare_parameter('input_image_topic', '/camera/color_image/compressed')
        self.declare_parameter('output_mask_topic', '/segmentation/color_mask/compressed')
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

        # --- LUT ---
        self.declare_parameter('lut_filename', 'color_lut.npy')
        self.declare_parameter('lut_dir', '')
        self.declare_parameter('smoothing_sigma', 14.0)
        self.declare_parameter('brush_size', 8)

        # --- LUT Confidence (per-label) ---
        self.declare_parameter('lut_confidence_sidewalk', 0.0)
        self.declare_parameter('lut_confidence_road', 0.0)
        self.declare_parameter('lut_confidence_lane', 0.0)

        # --- Morphological Cleanup ---
        self.declare_parameter('enable_morph_cleanup', True)
        self.declare_parameter('morph_kernel_size', 55)
        self.declare_parameter('lane_dilate_size', 20)

        # --- Edge Detection ---
        self.declare_parameter('enable_edge_detection', True)
        self.declare_parameter('edge_kernel_size', 13)

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
        self.brush_size = int(self.get_parameter('brush_size').value)
        self.lut_filename = str(self.get_parameter('lut_filename').value)
        input_topic = str(self.get_parameter('input_image_topic').value)
        output_topic = str(self.get_parameter('output_mask_topic').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.debug_logging = bool(self.get_parameter('debug_logging').value)
        self.smoothing_sigma = float(self.get_parameter('smoothing_sigma').value)

        clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        tile_size = int(self.get_parameter('clahe_tile_size').value)
        self.edge_kernel_size = int(self.get_parameter('edge_kernel_size').value)
        self.enable_edge_detection = bool(self.get_parameter('enable_edge_detection').value)

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
        self.lane_dilate_size = _odd_ksize(self.get_parameter('lane_dilate_size').value)

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

        # LUT confidence
        self.lut_confidence = [
            float(self.get_parameter('lut_confidence_sidewalk').value),
            float(self.get_parameter('lut_confidence_road').value),
            float(self.get_parameter('lut_confidence_lane').value),
        ]

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
        self.lut = None
        self.lut_gpu = None
        self.lut_max_prob = None
        self.lut_max_prob_gpu = None

        # Calibration state
        self.calibration_mode = False
        self.calibration_data = {cls: [] for cls in self.CLASS_NAMES}
        self.current_class_idx = 0
        self.calibration_frame_hsv = None
        self.calibration_frame_display = None
        self.mouse_pos = (0, 0)
        self._painting = False
        self._paint_mask = None
        self._stroke_history = []
        self.frame_count = 0

        # =============================================================
        # Resolve LUT config directory
        # =============================================================
        lut_dir_param = str(self.get_parameter('lut_dir').value)
        if lut_dir_param:
            self._config_dir = lut_dir_param
            self.get_logger().info(f"Using user-provided LUT dir: {self._config_dir}")
        else:
            source_config = self._find_source_config_dir()
            if source_config:
                self._config_dir = source_config
                self.get_logger().info(f"Auto-detected source config: {self._config_dir}")
            else:
                pkg_share = get_package_share_directory('qcar2_laneseg_acc')
                self._config_dir = os.path.join(pkg_share, 'config')
                self.get_logger().warn(
                    f"Could not find source config dir, using share: {self._config_dir}")

        self.lut_path = os.path.join(self._config_dir, self.lut_filename)

        if self._fisheye_enabled:
            self._load_fisheye_calibration()

        if self._load_lut():
            self.get_logger().info(f"LUT loaded from: {self.lut_path}")
            self._debug_analyze_lut()
        else:
            self.get_logger().warn("No LUT found. Calibration mode will start on first image.")
            self.calibration_mode = True

        # =============================================================
        # QoS + Pub/Sub
        # =============================================================
        qos_in = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_out = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.sub = self.create_subscription(
            CompressedImage, input_topic, self.image_callback, qos_in)
        self.pub = self.create_publisher(CompressedImage, output_topic, qos_out)
        self.undistorted_pub = self.create_publisher(
            CompressedImage, undistorted_topic, qos_out)

        # Startup log
        self.get_logger().info("Color Segmentation Node started")
        self.get_logger().info(f"  Input: {input_topic}")
        self.get_logger().info(f"  Output: {output_topic}")
        self.get_logger().info(f"  CuPy GPU: {'Enabled' if CUPY_AVAILABLE else 'Disabled'}")
        self.get_logger().info(f"  Pre-blur: {self.enable_pre_blur} (k={self.pre_blur_ksize})")
        self.get_logger().info(f"  Morph cleanup: {self.enable_morph_cleanup} (k={self.morph_kernel_size})")
        self.get_logger().info(f"  Adapt gamma: {self.enable_adapt_gamma}")
        self.get_logger().info(
            f"  AGCWD: mean_v=[{self.agcwd_mean_v_low},{self.agcwd_mean_v_high}] "
            f"dim_a={self.agcwd_dimmed_alpha} brt_a={self.agcwd_bright_alpha}")
        self.get_logger().info(
            f"  LUT confidence: sw={self.lut_confidence[0]} "
            f"rd={self.lut_confidence[1]} ln={self.lut_confidence[2]}")
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
            elif n == 'smoothing_sigma':      self.smoothing_sigma = float(v)
            elif n == 'debug_logging':        self.debug_logging = bool(v)
            elif n == 'jpeg_quality':         self.jpeg_quality = max(1, min(100, int(v)))
            elif n == 'brush_size':           self.brush_size = max(1, int(v))
            # CLAHE
            elif n == 'clahe_clip_limit':
                self.clahe = cv2.createCLAHE(
                    clipLimit=float(v),
                    tileGridSize=(int(self.get_parameter('clahe_tile_size').value),) * 2)
            elif n == 'clahe_tile_size':
                self.clahe = cv2.createCLAHE(
                    clipLimit=float(self.get_parameter('clahe_clip_limit').value),
                    tileGridSize=(int(v), int(v)))
            # Edge
            elif n == 'edge_kernel_size':        self.edge_kernel_size = int(v)
            elif n == 'enable_edge_detection':   self.enable_edge_detection = bool(v)
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
            elif n == 'lane_dilate_size':     self.lane_dilate_size = _odd_ksize(v)
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
            # LUT confidence
            elif n == 'lut_confidence_sidewalk':  self.lut_confidence[0] = float(v)
            elif n == 'lut_confidence_road':      self.lut_confidence[1] = float(v)
            elif n == 'lut_confidence_lane':      self.lut_confidence[2] = float(v)
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
    # LUT Management
    # =================================================================
    @property
    def _lut_probs_path(self):
        """Companion file for per-bin max probability."""
        base, ext = os.path.splitext(self.lut_path)
        return base + '_probs' + ext

    def _load_lut(self) -> bool:
        if os.path.exists(self.lut_path):
            try:
                self.lut = np.load(self.lut_path)
                if CUPY_AVAILABLE:
                    self.lut_gpu = cp.asarray(self.lut)
                # Load companion probability volume
                if os.path.exists(self._lut_probs_path):
                    self.lut_max_prob = np.load(self._lut_probs_path)
                    if CUPY_AVAILABLE:
                        self.lut_max_prob_gpu = cp.asarray(self.lut_max_prob)
                    self.get_logger().info("LUT probability volume loaded.")
                else:
                    self.get_logger().warn("No LUT probs file found — confidence filtering disabled.")
                    self.lut_max_prob = None
                    self.lut_max_prob_gpu = None
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to load LUT: {e}")
        return False

    def _save_lut(self) -> bool:
        try:
            config_dir = os.path.dirname(self.lut_path)
            os.makedirs(config_dir, exist_ok=True)
            np.save(self.lut_path, self.lut)
            if self.lut_max_prob is not None:
                np.save(self._lut_probs_path, self.lut_max_prob)
            self.get_logger().info(f"LUT saved to: {self.lut_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save LUT: {e}")
            return False

    def _generate_lut(self):
        self.get_logger().info("Generating LUT from calibration data...")
        self.get_logger().info(f"  Gaussian sigma={self.smoothing_sigma}")

        lut_shape = (180, 256, 256)
        histograms = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_hist = np.zeros(lut_shape, dtype=np.float32)
            total_pixels = 0
            for sample in self.calibration_data[class_name]:
                for h, s, v in sample:
                    class_hist[h, s, v] += 1
                    total_pixels += 1
            self.get_logger().info(
                f"  Class {class_idx} ({class_name}): {total_pixels} px")
            if total_pixels > 0:
                all_pixels = np.vstack(self.calibration_data[class_name])
                h_v, s_v, v_v = all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2]
                self.get_logger().info(
                    f"    HSV: H=[{h_v.min()}-{h_v.max()}] "
                    f"S=[{s_v.min()}-{s_v.max()}] V=[{v_v.min()}-{v_v.max()}]")
            if self.smoothing_sigma > 0 and total_pixels > 0:
                class_hist = gaussian_filter(class_hist, sigma=self.smoothing_sigma,
                                             mode='wrap')
            total = class_hist.sum()
            if total > 0:
                class_hist /= total
            histograms.append(class_hist)

        histograms = np.stack(histograms, axis=-1)  # (180,256,256,3)
        max_probs = np.max(histograms, axis=-1)      # (180,256,256)
        self.lut = np.argmax(histograms, axis=-1).astype(np.uint8)

        # Store per-bin max probability for confidence filtering
        self.lut_max_prob = max_probs.astype(np.float32)

        no_samples_mask = max_probs == 0
        self.lut[no_samples_mask] = 0
        self.lut_max_prob[no_samples_mask] = 0.0

        num_covered = (~no_samples_mask).sum()
        self.get_logger().info(
            f"  LUT coverage: {num_covered} bins "
            f"({100.0 * num_covered / self.lut.size:.2f}%)")

        if CUPY_AVAILABLE:
            self.lut_gpu = cp.asarray(self.lut)
            self.lut_max_prob_gpu = cp.asarray(self.lut_max_prob)

        self.get_logger().info("LUT generation complete!")
        self._debug_analyze_lut()

    # =================================================================
    # Image Processing Pipeline
    # =================================================================
    def _crop_roi(self, image):
        h, w = image.shape[:2]
        crop_start = int(h * (1 - self.roi_height_ratio))
        return image[crop_start:, :], crop_start

    def _apply_clahe(self, hsv_image):
        h, s, v = cv2.split(hsv_image)
        v_equalized = self.clahe.apply(v)
        return cv2.merge([h, s, v_equalized])

    def _segment_with_lut(self, hsv_image):
        h, s, v = cv2.split(hsv_image)

        if CUPY_AVAILABLE and self.lut_gpu is not None:
            h_g, s_g, v_g = cp.asarray(h), cp.asarray(s), cp.asarray(v)
            mask_gpu = self.lut_gpu[h_g, s_g, v_g]
            mask = cp.asnumpy(mask_gpu)
        else:
            mask = self.lut[h, s, v]

        # Per-label confidence filtering
        any_threshold = any(t > 0.0 for t in self.lut_confidence)
        if any_threshold and self.lut_max_prob is not None:
            if CUPY_AVAILABLE and self.lut_max_prob_gpu is not None:
                prob_gpu = self.lut_max_prob_gpu[h_g, s_g, v_g]
                prob = cp.asnumpy(prob_gpu)
            else:
                prob = self.lut_max_prob[h, s, v]

            for label_idx, threshold in enumerate(self.lut_confidence):
                if threshold > 0.0:
                    low_conf = (mask == label_idx) & (prob < threshold)
                    mask[low_conf] = 0  # Fallback to sidewalk

        return mask

    def _apply_robot_mask_output(self, full_mask: np.ndarray) -> np.ndarray:
        h, w = full_mask.shape[:2]
        x1 = max(0, min(self.robot_mask_x1, w))
        x2 = max(0, min(self.robot_mask_x2, w))
        y1 = max(0, min(self.robot_mask_y1, h))
        y2 = max(0, min(self.robot_mask_y2, h))
        if x2 > x1 and y2 > y1:
            full_mask[y1:y2, x1:x2] = 0
        return full_mask

    def _morph_cleanup(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size))
        road = (mask == 1).astype(np.uint8) * 255
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel, iterations=1)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel, iterations=1)
        road_cleaned = road > 0
        lane = (mask == 2).astype(np.uint8) * 255
        if self.lane_dilate_size > 1:
            lk = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.lane_dilate_size, self.lane_dilate_size))
            lane = cv2.dilate(lane, lk, iterations=1)
        lane_dilated = lane > 0
        result = np.zeros_like(mask)
        result[road_cleaned] = 1
        result[lane_dilated] = 2
        return result

    def _detect_road_edge(self, mask):
        road_mask = (mask == 1).astype(np.uint8) * 255
        sidewalk_mask = (mask == 0).astype(np.uint8) * 255
        ks = _odd_ksize(self.edge_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        road_dilated = cv2.dilate(road_mask, kernel, iterations=2)
        road_edge = cv2.bitwise_and(road_dilated, sidewalk_mask)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        road_edge = cv2.erode(road_edge, erosion_kernel, iterations=1)
        return road_edge > 0

    def _colorize_mask(self, mask):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in self.MASK_COLORS.items():
            colored[mask == label] = color
        return colored

    # =================================================================
    # Calibration GUI (Paint-based)
    # =================================================================
    def _collect_painted_pixels(self, x, y):
        if self.calibration_frame_hsv is None or self._paint_mask is None:
            return
        h, w = self.calibration_frame_hsv.shape[:2]
        stamp = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(stamp, (x, y), self.brush_size, 255, -1)
        new_pixels = (stamp > 0) & (self._paint_mask == 0)
        if not np.any(new_pixels):
            return
        hsv_pixels = self.calibration_frame_hsv[new_pixels]
        current_class = self.CLASS_NAMES[self.current_class_idx]
        self.calibration_data[current_class].append(hsv_pixels)

    def _mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._painting = True
            if self._paint_mask is not None:
                self._stroke_history.append(self._paint_mask.copy())
            self._collect_painted_pixels(x, y)
            if self._paint_mask is not None:
                cv2.circle(self._paint_mask, (x, y), self.brush_size, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and self._painting:
            self._collect_painted_pixels(x, y)
            if self._paint_mask is not None:
                cv2.circle(self._paint_mask, (x, y), self.brush_size, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self._painting = False

    def _advance_class(self):
        current_class = self.CLASS_NAMES[self.current_class_idx]
        total_px = sum(len(s) for s in self.calibration_data[current_class])
        if total_px == 0:
            self.get_logger().warn("No pixels painted — paint a region first.")
            return
        self.get_logger().info(f"Class '{current_class}' done: {total_px} px collected.")
        self.current_class_idx += 1
        self._paint_mask[:] = 0
        self._stroke_history.clear()
        if self.current_class_idx >= len(self.CLASS_NAMES):
            self._finish_calibration()

    def _finish_calibration(self):
        self.get_logger().info("Calibration complete! Generating LUT...")
        cv2.destroyAllWindows()
        self._generate_lut()
        self._save_lut()
        self.calibration_mode = False
        self._paint_mask = None
        self._stroke_history.clear()
        self.get_logger().info("Ready for segmentation!")

    def _run_calibration_gui(self, roi_bgr, hsv_normalized):
        self.calibration_frame_hsv = hsv_normalized.copy()
        self.calibration_frame_display = roi_bgr.copy()
        h, w = roi_bgr.shape[:2]
        if self._paint_mask is None or self._paint_mask.shape[:2] != (h, w):
            self._paint_mask = np.zeros((h, w), dtype=np.uint8)
            self._stroke_history.clear()
        display = roi_bgr.copy()
        if self._paint_mask is not None:
            current_class = self.CLASS_NAMES[self.current_class_idx]
            overlay_color = self.MASK_COLORS.get(self.current_class_idx, (0, 255, 0))
            if current_class == 'sidewalk':
                overlay_color = (180, 180, 180)
            overlay = display.copy()
            overlay[self._paint_mask > 0] = overlay_color
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)
        x, y = self.mouse_pos
        cv2.circle(display, (x, y), self.brush_size, (0, 255, 0), 1)
        current_class = self.CLASS_NAMES[self.current_class_idx]
        color = self.MASK_COLORS[self.current_class_idx]
        text_color = (255, 255, 255) if current_class == 'sidewalk' else color
        num_painted = int((self._paint_mask > 0).sum()) if self._paint_mask is not None else 0
        cv2.putText(display, f"Paint: {current_class.upper()}  ({num_painted} px)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(display, "ENTER=confirm | c=clear | u=undo | +/-=brush | q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(display,
                    f"Class {self.current_class_idx + 1}/{len(self.CLASS_NAMES)}  "
                    f"brush={self.brush_size}px",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Calibration", display)
        cv2.setMouseCallback("Calibration", self._mouse_callback)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            self._advance_class()
        elif key == ord('c'):
            current_class = self.CLASS_NAMES[self.current_class_idx]
            total_px = sum(len(s) for s in self.calibration_data[current_class])
            self._paint_mask[:] = 0
            self._stroke_history.clear()
            self.get_logger().info(
                f"Canvas cleared — {total_px} px saved for '{current_class}'.")
        elif key == ord('u'):
            if self._stroke_history:
                self._paint_mask = self._stroke_history.pop()
                self.get_logger().info("Undo stroke.")
            else:
                self.get_logger().info("Nothing to undo.")
        elif key == ord('+') or key == ord('='):
            self.brush_size = min(100, self.brush_size + 2)
        elif key == ord('-'):
            self.brush_size = max(1, self.brush_size - 2)
        elif key == ord('q'):
            self.get_logger().warn("Calibration cancelled")
            cv2.destroyAllWindows()
            rclpy.shutdown()

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

            # 3) HSV + CLAHE
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_normalized = self._apply_clahe(hsv)

            if self.calibration_mode:
                self._run_calibration_gui(roi, hsv_normalized)
                return

            self.frame_count += 1
            if self.debug_logging and self.frame_count % 30 == 0:
                self._debug_log_hsv_stats(hsv_normalized, "Inference")

            # 4) LUT segmentation (with confidence filtering)
            mask = self._segment_with_lut(hsv_normalized)

            if self.debug_logging and self.frame_count % 30 == 0:
                self._debug_log_mask_stats(mask)

            # 5) Morphological cleanup
            if self.enable_morph_cleanup:
                mask = self._morph_cleanup(mask)

            # 6) Edge detection
            if self.enable_edge_detection:
                road_edge = self._detect_road_edge(mask)
                mask[road_edge] = 3

            # 7) Colorize
            colored_mask = self._colorize_mask(mask)
            full_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            full_mask[crop_offset:, :] = colored_mask

            # 8) Robot mask
            if self.enable_robot_mask:
                full_mask = self._apply_robot_mask_output(full_mask)

            # 9) Publish
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            _, compressed_data = cv2.imencode('.jpg', full_mask, encode_params)
            out_msg = CompressedImage()
            out_msg.header = msg.header
            out_msg.format = 'jpeg'
            out_msg.data = compressed_data.tobytes()
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

    # =================================================================
    # Debug
    # =================================================================
    def _debug_analyze_lut(self):
        if self.lut is None:
            return
        self.get_logger().info("=== LUT Analysis ===")
        self.get_logger().info(f"  Shape: {self.lut.shape}")
        unique, counts = np.unique(self.lut, return_counts=True)
        total_bins = self.lut.size
        for label, count in zip(unique, counts):
            pct = 100.0 * count / total_bins
            cn = self.CLASS_NAMES[label] if label < len(self.CLASS_NAMES) else f"unknown_{label}"
            self.get_logger().info(f"  Class {label} ({cn}): {count} bins ({pct:.2f}%)")

    def _debug_log_hsv_stats(self, hsv_image, context=""):
        if not self.debug_logging:
            return
        h, s, v = cv2.split(hsv_image)
        self.get_logger().info(
            f"[{context}] HSV: H=[{h.min()}-{h.max()}] "
            f"S=[{s.min()}-{s.max()}] V=[{v.min()}-{v.max()}]")

    def _debug_log_mask_stats(self, mask):
        if not self.debug_logging:
            return
        unique, counts = np.unique(mask, return_counts=True)
        stats = ", ".join([f"{l}:{c}" for l, c in zip(unique, counts)])
        self.get_logger().info(f"[Mask] Labels: {stats}")


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
