#!/usr/bin/env python3
"""
HSV Color Segmentation Node with CuPy GPU Acceleration

This node performs real-time semantic segmentation using histogram-based
color classification in HSV space with GPU acceleration via CuPy.

Pipeline:
1. ROI Crop - Keep bottom portion of image
2. (Optional) Gaussian Blur - Reduce noise before HSV conversion
3. HSV + CLAHE - Convert to HSV and normalize illumination
4. LUT Segmentation - Apply precomputed lookup table
5. (Optional) Morphological Cleanup - Close + Open to remove noise/holes
6. Edge Detection - Find border between sidewalk and road
7. Colorize mask
8. (Optional) Robot Mask - Zero-fill rectangle in final output (like ROI crop)
9. Publish segmentation mask

Author: hackbrian (+ improvements Eduardex)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from scipy.ndimage import gaussian_filter

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[WARN] CuPy not available, falling back to NumPy (CPU)")


def _odd_ksize(k: int) -> int:
    k = int(k)
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else (k + 1)


# =========================================================================
# Adaptive Gamma Correction (AGCWD) — from adaptgamma.py
# =========================================================================
def _image_agcwd(img_u8: np.ndarray, a: float = 0.25, truncated_cdf: bool = False) -> np.ndarray:
    """AGCWD sobre imagen uint8 de un canal."""
    if img_u8.dtype != np.uint8:
        img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

    hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return img_u8.copy()

    prob = hist / total
    prob_min = prob.min()
    prob_max = prob.max()
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

    if truncated_cdf:
        inverse_cdf = np.maximum(0.5, 1.0 - cdf_wd)
    else:
        inverse_cdf = 1.0 - cdf_wd

    lut = np.arange(256, dtype=np.float64)
    lut = np.round(255.0 * np.power(lut / 255.0, inverse_cdf))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img_u8, lut)


def _process_bright(img_u8: np.ndarray, a_val: float = 0.6) -> np.ndarray:
    """Para imágenes muy brillantes."""
    img_negative = 255 - img_u8
    agcwd_neg = _image_agcwd(img_negative, a=a_val, truncated_cdf=False)
    return 255 - agcwd_neg


def _process_dimmed(img_u8: np.ndarray, a_val: float = 0.75) -> np.ndarray:
    """Para imágenes oscuras."""
    return _image_agcwd(img_u8, a=a_val, truncated_cdf=True)


def _compress_highlights(v_channel: np.ndarray, sat_mask: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Comprime zonas sobreexpuestas localmente."""
    v = v_channel.astype(np.float32)
    out = v.copy()
    norm = v / 255.0
    out[sat_mask > 0] = 255.0 * np.power(norm[sat_mask > 0], 1.0 / max(strength, 1e-3))
    return np.clip(out, 0, 255).astype(np.uint8)


def _correct_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Corrección robusta de iluminación:
    - trabaja sobre luminancia V en HSV
    - detecta highlights
    - aplica AGCWD según iluminación global
    - comprime highlights localmente
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_v = float(np.mean(v))
    p95_v = float(np.percentile(v, 95))
    sat_ratio = float(np.mean(v > 245))

    # Máscara de sobreexposición
    sat_mask = ((v > 240) & (s < 80)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)
    sat_mask = cv2.medianBlur(sat_mask, 5)

    if mean_v < 110:
        v_corr = _process_dimmed(v, a_val=0.85)
        s_corr = cv2.convertScaleAbs(s, alpha=1.5, beta=10)
    elif mean_v > 130 or p95_v > 240 or sat_ratio > 0.02:
        v_corr = _process_bright(v, a_val=0.70)
        v_corr = _compress_highlights(v_corr, sat_mask, strength=0.45)
        s_corr = cv2.convertScaleAbs(s, alpha=2.0, beta=15)
    else:
        v_corr = v.copy()
        s_corr = cv2.convertScaleAbs(s, alpha=1.4, beta=5)

    hsv_corr = cv2.merge([h, s_corr, v_corr])
    return cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)


class ColorSegmentationNode(Node):
    """
    ROS2 Node for HSV-based color segmentation with GUI calibration.

    Output Labels:
        0 - Sidewalk (Black mask)
        1 - Road (Blue mask)
        2 - Lane (Yellow mask)
        3 - Road Edge (Red mask)
    """

    # Mask colors in BGR format
    MASK_COLORS = {
        0: (0, 0, 0),       # Sidewalk - Black
        1: (255, 0, 0),     # Road - Blue
        2: (0, 255, 255),   # Lane - Yellow
        3: (0, 0, 255),     # Road Edge - Red
    }

    CLASS_NAMES = ['sidewalk', 'road', 'lane']

    def __init__(self):
        super().__init__('color_segmentation_node')

        # =================================================================
        # Parameters IR JUGANDO CON ESTOS VALORES
        # =================================================================
        self.declare_parameter('roi_height_ratio', 0.2) 
        self.declare_parameter('brush_size', 8)  # Radius of paint brush for calibration
        self.declare_parameter('lut_filename', 'color_lut.npy')
        self.declare_parameter('input_image_topic', '/camera/color_image/compressed')
        self.declare_parameter('output_mask_topic', '/segmentation/color_mask/compressed')
        self.declare_parameter('jpeg_quality', 10)  # 1-100, lower = more compression
        self.declare_parameter('clahe_clip_limit', 1.05) #0.5 1.05
        self.declare_parameter('clahe_tile_size', 5) #1
        self.declare_parameter('edge_kernel_size', 9)
        self.declare_parameter('enable_edge_detection', True)
        self.declare_parameter('debug_logging', True)
        self.declare_parameter('smoothing_sigma', 14.0)
        self.declare_parameter('lut_dir', '')  # Override: path to config dir for LUT

        # =================================================================
        # NEW: Pre-blur (reduce ruido antes de HSV)
        # =================================================================
        self.declare_parameter('enable_pre_blur', True)
        self.declare_parameter('pre_blur_ksize', 9)     # impar (aumentado para reducir oscilación)
        self.declare_parameter('pre_blur_sigma', 0.0)   # 0 = auto

        # =================================================================
        # NEW: Robot mask (hide robot parts visible in image)
        # =================================================================
        self.declare_parameter('enable_robot_mask', False)
        self.declare_parameter('robot_mask_x1', 400)      # Top-left X
        self.declare_parameter('robot_mask_y1', 430)      # Top-left Y
        self.declare_parameter('robot_mask_x2', 530)      # Bottom-right X
        self.declare_parameter('robot_mask_y2', 480)      # Bottom-right Y

        # =================================================================
        # NEW: Morphological cleanup (Close + Open operations)
        # =================================================================
        self.declare_parameter('enable_morph_cleanup', True)
        self.declare_parameter('morph_kernel_size', 55)  #65 Kernel size for morphological ops
        self.declare_parameter('lane_dilate_size', 5)   # Lane dilation kernel size (0=disabled)

        # =================================================================
        # NEW: Adaptive Gamma Correction (AGCWD) preprocessing
        # =================================================================
        self.declare_parameter('enable_adapt_gamma', True)

        # Get parameters
        self.roi_height_ratio = float(self.get_parameter('roi_height_ratio').value)
        self.brush_size = int(self.get_parameter('brush_size').value)
        self.lut_filename = str(self.get_parameter('lut_filename').value)
        input_topic = str(self.get_parameter('input_image_topic').value)
        output_topic = str(self.get_parameter('output_mask_topic').value)
        clip_limit = float(self.get_parameter('clahe_clip_limit').value)
        tile_size = int(self.get_parameter('clahe_tile_size').value)
        self.edge_kernel_size = int(self.get_parameter('edge_kernel_size').value)
        self.enable_edge_detection = bool(self.get_parameter('enable_edge_detection').value)
        self.debug_logging = bool(self.get_parameter('debug_logging').value)
        self.smoothing_sigma = float(self.get_parameter('smoothing_sigma').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        # New params
        self.enable_pre_blur = bool(self.get_parameter('enable_pre_blur').value)
        self.pre_blur_ksize = _odd_ksize(self.get_parameter('pre_blur_ksize').value)
        self.pre_blur_sigma = float(self.get_parameter('pre_blur_sigma').value)

        # Robot mask params
        self.enable_robot_mask = bool(self.get_parameter('enable_robot_mask').value)
        self.robot_mask_x1 = int(self.get_parameter('robot_mask_x1').value)
        self.robot_mask_y1 = int(self.get_parameter('robot_mask_y1').value)
        self.robot_mask_x2 = int(self.get_parameter('robot_mask_x2').value)
        self.robot_mask_y2 = int(self.get_parameter('robot_mask_y2').value)

        # Morphological cleanup params
        self.enable_morph_cleanup = bool(self.get_parameter('enable_morph_cleanup').value)
        self.morph_kernel_size = _odd_ksize(self.get_parameter('morph_kernel_size').value)

        # Adaptive gamma correction
        self.enable_adapt_gamma = bool(self.get_parameter('enable_adapt_gamma').value)
        self.lane_dilate_size = _odd_ksize(self.get_parameter('lane_dilate_size').value)

        # =================================================================
        # Initialize components
        # =================================================================
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        # LUT storage (will be loaded or generated)
        self.lut = None
        self.lut_gpu = None

        # Calibration state
        self.calibration_mode = False
        self.calibration_data = {cls: [] for cls in self.CLASS_NAMES}
        self.current_class_idx = 0
        self.calibration_frame_hsv = None
        self.calibration_frame_display = None
        self.mouse_pos = (0, 0)
        self._painting = False          # True while left-button is held
        self._paint_mask = None         # Binary mask of painted pixels
        self._stroke_history = []       # List of masks for undo support

        # Debug counter
        self.frame_count = 0

        # =================================================================
        # Resolve LUT config directory (source > share fallback)
        # =================================================================
        lut_dir_param = str(self.get_parameter('lut_dir').value)

        if lut_dir_param:
            # User explicitly provided the config directory
            self._config_dir = lut_dir_param
            self.get_logger().info(f"Using user-provided LUT dir: {self._config_dir}")
        else:
            # Auto-detect source config directory from colcon workspace
            source_config = self._find_source_config_dir()
            if source_config:
                self._config_dir = source_config
                self.get_logger().info(f"Auto-detected source config: {self._config_dir}")
            else:
                # Fallback to install share directory
                pkg_share = get_package_share_directory('qcar2_laneseg_acc')
                self._config_dir = os.path.join(pkg_share, 'config')
                self.get_logger().warn(
                    f"Could not find source config dir, using share: {self._config_dir}"
                )

        self.lut_path = os.path.join(self._config_dir, self.lut_filename)

        # Try to load existing LUT
        if self._load_lut():
            self.get_logger().info(f"LUT loaded from: {self.lut_path}")
            self._debug_analyze_lut()
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
        self.sub = self.create_subscription(CompressedImage, input_topic, self.image_callback, qos_input)
        self.pub = self.create_publisher(CompressedImage, output_topic, qos_output)

        self.get_logger().info("Color Segmentation Node started")
        self.get_logger().info(f"  Input: {input_topic}")
        self.get_logger().info(f"  Output: {output_topic}")
        self.get_logger().info(f"  CuPy GPU: {'Enabled' if CUPY_AVAILABLE else 'Disabled (CPU fallback)'}")
        self.get_logger().info(f"  Debug logging: {self.debug_logging}")
        self.get_logger().info(f"  Edge detection: {self.enable_edge_detection}")
        self.get_logger().info(f"  Pre-blur: {self.enable_pre_blur} (k={self.pre_blur_ksize}, sigma={self.pre_blur_sigma})")
        self.get_logger().info(f"  Robot mask: {self.enable_robot_mask} (rect=[{self.robot_mask_x1},{self.robot_mask_y1}]-[{self.robot_mask_x2},{self.robot_mask_y2}])")
        self.get_logger().info(f"  Morph cleanup: {self.enable_morph_cleanup} (kernel={self.morph_kernel_size})")
        self.get_logger().info(f"  Adapt gamma: {self.enable_adapt_gamma}")

        # =================================================================
        # Dynamic Parameter Reconfiguration
        # =================================================================
        self.add_on_set_parameters_callback(self._parameter_callback)

    # =====================================================================
    # Dynamic Parameter Callback
    # =====================================================================
    def _parameter_callback(self, params):
        """
        Called whenever a parameter is changed at runtime (e.g. via rqt).
        Updates the corresponding instance variable so the pipeline uses
        the new value on the next frame.  Topic / LUT-path parameters are
        read-only at runtime because they require a full restart.
        """
        for param in params:
            name = param.name
            value = param.value

            # --- ROI / General ---
            if name == 'roi_height_ratio':
                self.roi_height_ratio = float(value)
            elif name == 'smoothing_sigma':
                self.smoothing_sigma = float(value)
            elif name == 'debug_logging':
                self.debug_logging = bool(value)

            # --- CLAHE (needs object recreation) ---
            elif name == 'clahe_clip_limit':
                clip = float(value)
                tile = int(self.get_parameter('clahe_tile_size').value)
                self.clahe = cv2.createCLAHE(
                    clipLimit=clip, tileGridSize=(tile, tile)
                )
            elif name == 'clahe_tile_size':
                tile = int(value)
                clip = float(self.get_parameter('clahe_clip_limit').value)
                self.clahe = cv2.createCLAHE(
                    clipLimit=clip, tileGridSize=(tile, tile)
                )

            # --- Edge detection ---
            elif name == 'edge_kernel_size':
                self.edge_kernel_size = int(value)
            elif name == 'enable_edge_detection':
                self.enable_edge_detection = bool(value)

            # --- Pre-blur ---
            elif name == 'enable_pre_blur':
                self.enable_pre_blur = bool(value)
            elif name == 'pre_blur_ksize':
                self.pre_blur_ksize = _odd_ksize(value)
            elif name == 'pre_blur_sigma':
                self.pre_blur_sigma = float(value)

            # --- Robot mask ---
            elif name == 'enable_robot_mask':
                self.enable_robot_mask = bool(value)
            elif name == 'robot_mask_x1':
                self.robot_mask_x1 = int(value)
            elif name == 'robot_mask_y1':
                self.robot_mask_y1 = int(value)
            elif name == 'robot_mask_x2':
                self.robot_mask_x2 = int(value)
            elif name == 'robot_mask_y2':
                self.robot_mask_y2 = int(value)

            # --- Morphological cleanup ---
            elif name == 'enable_morph_cleanup':
                self.enable_morph_cleanup = bool(value)
            elif name == 'morph_kernel_size':
                self.morph_kernel_size = _odd_ksize(value)
            elif name == 'lane_dilate_size':
                self.lane_dilate_size = _odd_ksize(value)

            # --- Adaptive gamma ---
            elif name == 'enable_adapt_gamma':
                self.enable_adapt_gamma = bool(value)

            # --- Output compression ---
            elif name == 'jpeg_quality':
                self.jpeg_quality = max(1, min(100, int(value)))

            # --- Calibration ---
            elif name == 'brush_size':
                self.brush_size = max(1, int(value))

            self.get_logger().info(f"Parameter '{name}' updated to: {value}")

        return SetParametersResult(successful=True)

    # =====================================================================
    # Source Directory Auto-Detection
    # =====================================================================
    def _find_source_config_dir(self):
        """
        Auto-detect the source config/ directory from the colcon workspace.

        Strategy:
          1. Get installed share path via get_package_share_directory()
             e.g. <ws>/install/<pkg>/share/<pkg>
          2. Go up 4 levels to reach the workspace root <ws>/
          3. Scan <ws>/src/ for a package.xml containing our package name
          4. Return <source_pkg_dir>/config/
        """
        try:
            share_dir = get_package_share_directory('qcar2_laneseg_acc')
            # share_dir = <ws>/install/<pkg>/share/<pkg>
            # Navigate up 4 levels: share/<pkg> -> share -> <pkg> -> install -> <ws>
            ws_root = share_dir
            for _ in range(4):
                ws_root = os.path.dirname(ws_root)

            src_dir = os.path.join(ws_root, 'src')
            if not os.path.isdir(src_dir):
                return None

            # Walk src/ (max depth 2) looking for our package.xml
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

    # =====================================================================
    # Debug Functions
    # =====================================================================
    def _debug_analyze_lut(self):
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
        if not self.debug_logging:
            return
        h, s, v = cv2.split(hsv_image)
        self.get_logger().info(
            f"[{context}] HSV stats: H=[{h.min()}-{h.max()}], "
            f"S=[{s.min()}-{s.max()}], V=[{v.min()}-{v.max()}]"
        )

    def _debug_log_mask_stats(self, mask):
        if not self.debug_logging:
            return
        unique, counts = np.unique(mask, return_counts=True)
        stats = ", ".join([f"{label}:{count}" for label, count in zip(unique, counts)])
        self.get_logger().info(f"[Mask] Labels: {stats}")

    # =====================================================================
    # LUT Management
    # =====================================================================
    def _load_lut(self) -> bool:
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
        try:
            config_dir = os.path.dirname(self.lut_path)
            os.makedirs(config_dir, exist_ok=True)
            np.save(self.lut_path, self.lut)
            self.get_logger().info(f"LUT saved to: {self.lut_path}")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to save LUT: {e}")
            return False

    def _generate_lut(self):
        self.get_logger().info("Generating LUT from calibration data...")
        self.get_logger().info(f"  Using Gaussian smoothing with sigma={self.smoothing_sigma}")

        lut_shape = (180, 256, 256)
        histograms = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_hist = np.zeros(lut_shape, dtype=np.float32)
            total_pixels = 0

            for sample in self.calibration_data[class_name]:
                for h, s, v in sample:
                    class_hist[h, s, v] += 1
                    total_pixels += 1

            self.get_logger().info(f"  Class {class_idx} ({class_name}): {total_pixels} total pixels sampled")

            if total_pixels > 0:
                all_pixels = np.vstack(self.calibration_data[class_name])
                h_vals, s_vals, v_vals = all_pixels[:, 0], all_pixels[:, 1], all_pixels[:, 2]
                self.get_logger().info(
                    f"    HSV ranges: H=[{h_vals.min()}-{h_vals.max()}], "
                    f"S=[{s_vals.min()}-{s_vals.max()}], V=[{v_vals.min()}-{v_vals.max()}]"
                )

            if self.smoothing_sigma > 0 and total_pixels > 0:
                class_hist = gaussian_filter(
                    class_hist,
                    sigma=self.smoothing_sigma,
                    mode='wrap'
                )
                self.get_logger().info(f"    Applied Gaussian smoothing (sigma={self.smoothing_sigma})")

            total = class_hist.sum()
            if total > 0:
                class_hist /= total

            histograms.append(class_hist)

        histograms = np.stack(histograms, axis=-1)
        max_probs = np.max(histograms, axis=-1)
        self.lut = np.argmax(histograms, axis=-1).astype(np.uint8)

        has_samples_mask = max_probs > 0
        num_covered = has_samples_mask.sum()
        coverage_pct = 100.0 * num_covered / self.lut.size
        self.get_logger().info(f"  LUT coverage after smoothing: {num_covered} bins ({coverage_pct:.2f}%)")

        no_samples_mask = max_probs == 0
        num_empty = no_samples_mask.sum()
        self.get_logger().info(f"  Empty bins (no influence): {num_empty} ({100.0*num_empty/self.lut.size:.2f}%)")
        self.lut[no_samples_mask] = 0

        if CUPY_AVAILABLE:
            self.lut_gpu = cp.asarray(self.lut)

        self.get_logger().info("LUT generation complete!")
        self._debug_analyze_lut()

    # =====================================================================
    # Image Processing Pipeline
    # =====================================================================
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
            h_gpu = cp.asarray(h)
            s_gpu = cp.asarray(s)
            v_gpu = cp.asarray(v)
            mask_gpu = self.lut_gpu[h_gpu, s_gpu, v_gpu]
            mask = cp.asnumpy(mask_gpu)
        else:
            mask = self.lut[h, s, v]

        return mask

    def _apply_robot_mask_output(self, full_mask: np.ndarray) -> np.ndarray:
        """
        Zero-fill a rectangular region in the final output mask.
        The rectangle is defined by two corner points: (x1,y1) to (x2,y2).
        Coordinates are in GLOBAL image space (original image).
        This completely ignores the region, similar to how ROI crop ignores the top.
        """
        h, w = full_mask.shape[:2]
        
        # Clamp to image bounds (using global coordinates directly)
        x1 = max(0, min(self.robot_mask_x1, w))
        x2 = max(0, min(self.robot_mask_x2, w))
        y1 = max(0, min(self.robot_mask_y1, h))
        y2 = max(0, min(self.robot_mask_y2, h))
        
        if x2 > x1 and y2 > y1:
            full_mask[y1:y2, x1:x2] = 0  # Set to black (ignored)
        
        return full_mask

    def _morph_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the segmentation mask.
        Close (fill holes) + Open (remove noise) on the road class.
        Dilate lanes to make them thicker.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # Process road class (label=1)
        road = (mask == 1).astype(np.uint8) * 255
        # Close: fill small holes in road
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Open: remove small noise/islands
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel, iterations=1)
        road_cleaned = road > 0
        
        # Process lane class (label=2) - dilate to make thicker
        lane = (mask == 2).astype(np.uint8) * 255
        if self.lane_dilate_size > 1:
            lane_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.lane_dilate_size, self.lane_dilate_size)
            )
            lane = cv2.dilate(lane, lane_kernel, iterations=1)
        lane_dilated = lane > 0
        
        # Final mask: sidewalk (0) by default, then overlay road and lanes
        # Lanes have priority over road
        result = np.zeros_like(mask)
        result[road_cleaned] = 1
        result[lane_dilated] = 2
        
        return result

    def _detect_road_edge(self, mask):
        """
        Detect edge between sidewalk (0) and road (1).
        Border = dilate(road) ∩ sidewalk
        """
        road_mask = (mask == 1).astype(np.uint8) * 255
        sidewalk_mask = (mask == 0).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_odd_ksize(self.edge_kernel_size), _odd_ksize(self.edge_kernel_size))
        )
        road_dilated = cv2.dilate(road_mask, kernel, iterations=2)
        road_edge = cv2.bitwise_and(road_dilated, sidewalk_mask)
        
        # Erosion to remove loose/oscillating edge pixels (use smaller kernel)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        road_edge = cv2.erode(road_edge, erosion_kernel, iterations=1)

        return road_edge > 0

    def _colorize_mask(self, mask):
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in self.MASK_COLORS.items():
            colored[mask == label] = color
        return colored

    # =====================================================================
    # Calibration GUI  (Paint-based)
    # =====================================================================
    def _collect_painted_pixels(self, x, y):
        """Collect HSV values under the brush circle and add them immediately."""
        if self.calibration_frame_hsv is None or self._paint_mask is None:
            return
        h, w = self.calibration_frame_hsv.shape[:2]
        # Build a temporary mask for just this brush stamp
        stamp = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(stamp, (x, y), self.brush_size, 255, -1)
        # Only collect NEW pixels (not already painted)
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
            # Start a new stroke: save a snapshot of the current mask
            if self._paint_mask is not None:
                self._stroke_history.append(self._paint_mask.copy())
            # Collect data & paint first dot
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
        """Advance to the next class (ENTER key). Data was already collected while painting."""
        current_class = self.CLASS_NAMES[self.current_class_idx]
        total_px = sum(len(s) for s in self.calibration_data[current_class])

        if total_px == 0:
            self.get_logger().warn("No pixels painted — paint a region first.")
            return

        self.get_logger().info(
            f"Class '{current_class}' done: {total_px} px collected."
        )

        # Advance to the next class
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

        # Lazy-init paint mask to image size
        if self._paint_mask is None or self._paint_mask.shape[:2] != (h, w):
            self._paint_mask = np.zeros((h, w), dtype=np.uint8)
            self._stroke_history.clear()

        # --- Build display ---
        display = roi_bgr.copy()

        # Draw paint overlay (semi-transparent highlight)
        if self._paint_mask is not None:
            current_class = self.CLASS_NAMES[self.current_class_idx]
            overlay_color = self.MASK_COLORS.get(self.current_class_idx, (0, 255, 0))
            # White overlay for sidewalk (black class) so the paint is visible
            if current_class == 'sidewalk':
                overlay_color = (180, 180, 180)
            overlay = display.copy()
            overlay[self._paint_mask > 0] = overlay_color
            cv2.addWeighted(overlay, 0.45, display, 0.55, 0, display)

        # Draw brush cursor
        x, y = self.mouse_pos
        cv2.circle(display, (x, y), self.brush_size, (0, 255, 0), 1)

        # --- HUD text ---
        current_class = self.CLASS_NAMES[self.current_class_idx]
        color = self.MASK_COLORS[self.current_class_idx]
        text_color = (255, 255, 255) if current_class == 'sidewalk' else color
        num_painted = int((self._paint_mask > 0).sum()) if self._paint_mask is not None else 0

        cv2.putText(display,
                    f"Paint: {current_class.upper()}  ({num_painted} px)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(display,
                    "ENTER=confirm | c=clear | u=undo | +/-=brush | q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(display,
                    f"Class {self.current_class_idx + 1}/{len(self.CLASS_NAMES)}  "
                    f"brush={self.brush_size}px",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Calibration", display)
        cv2.setMouseCallback("Calibration", self._mouse_callback)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER — advance to next class
            self._advance_class()
        elif key == ord('c'):  # Clear canvas, keep data, continue same class
            current_class = self.CLASS_NAMES[self.current_class_idx]
            total_px = sum(len(s) for s in self.calibration_data[current_class])
            self._paint_mask[:] = 0
            self._stroke_history.clear()
            self.get_logger().info(
                f"Canvas cleared — {total_px} px saved for '{current_class}'. "
                f"Keep painting to add more."
            )
        elif key == ord('u'):  # Undo last stroke
            if self._stroke_history:
                self._paint_mask = self._stroke_history.pop()
                self.get_logger().info("Undo stroke.")
            else:
                self.get_logger().info("Nothing to undo.")
        elif key == ord('+') or key == ord('='):  # Increase brush
            self.brush_size = min(100, self.brush_size + 2)
        elif key == ord('-'):  # Decrease brush
            self.brush_size = max(1, self.brush_size - 2)
        elif key == ord('q'):
            self.get_logger().warn("Calibration cancelled")
            cv2.destroyAllWindows()
            rclpy.shutdown()

    # =====================================================================
    # Main Callback
    # =====================================================================
    def image_callback(self, msg):
        try:
            # Decode compressed image
            if not msg.data:
                return
            np_arr = np.frombuffer(msg.data, np.uint8)
            if np_arr.size == 0:
                return
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return

            # 0) Adaptive Gamma Correction — normalize illumination
            if self.enable_adapt_gamma:
                frame = _correct_frame(frame)

            roi, crop_offset = self._crop_roi(frame)

            # 1) Pre-blur: reduce noise
            if self.enable_pre_blur and self.pre_blur_ksize > 1:
                roi = cv2.GaussianBlur(roi, (self.pre_blur_ksize, self.pre_blur_ksize), self.pre_blur_sigma)

            # 2) HSV conversion
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 3) CLAHE normalization
            hsv_normalized = self._apply_clahe(hsv)

            if self.calibration_mode:
                self._run_calibration_gui(roi, hsv_normalized)
                return

            self.frame_count += 1
            if self.debug_logging and self.frame_count % 30 == 0:
                self._debug_log_hsv_stats(hsv_normalized, "Inference")

            # 4) LUT segmentation
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

            # 8) Robot mask: zero-fill rectangle region (applied to final output)
            if self.enable_robot_mask:
                full_mask = self._apply_robot_mask_output(full_mask)

            # 9) Publish as CompressedImage (JPEG, max compression)
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
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
