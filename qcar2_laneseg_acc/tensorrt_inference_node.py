#!/usr/bin/env python3
"""
TensorRT Inference Node (Standalone - No NITROS)

Alternativa al isaac_ros_tensor_rt que usa TensorRT directamente.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

class TensorRTInferenceNode(Node):
    def __init__(self):
        super().__init__('tensorrt_inference_node')
        
        self.declare_parameter('engine_file', '')
        self.declare_parameter('input_topic', '/camera/color_image')
        self.declare_parameter('output_topic', '/segmentation/mask')
        
        engine_file = self.get_parameter('engine_file').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        self.bridge = CvBridge()
        
        # Load TensorRT engine
        self.get_logger().info(f'Loading TensorRT engine: {engine_file}')
        self.engine, self.context = self.load_engine(engine_file)
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # Subscribers and publishers
        self.sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Image, output_topic, 10)
        
        self.get_logger().info('TensorRT Inference Node initialized')
    
    def load_engine(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess: resize to 224x224, normalize
            input_image = cv2.resize(cv_image, (224, 224))
            input_image = input_image.astype(np.float32) / 255.0
            
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_image = (input_image - mean) / std
            
            # HWC -> CHW
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
            
            # Copy to input buffer
            np.copyto(self.inputs[0]['host'], input_image.ravel())
            
            # Transfer input data to GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Run inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer predictions back
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            
            # Synchronize
            self.stream.synchronize()
            
            # Reshape output (1, 1, 224, 224)
            output = self.outputs[0]['host'].reshape(1, 1, 224, 224)
            mask = output[0, 0].astype(np.uint8)
            
            # Publish mask
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header
            self.pub.publish(mask_msg)
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = TensorRTInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
