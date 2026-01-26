cmake_minimum_required(VERSION 3.8)
project(qcar2_laneseg_acc)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# Find dependencies
# =============================================================================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# =============================================================================
# Python Module Installation
# =============================================================================
# This installs the Python package (qcar2_laneseg_acc/) as an importable module
ament_python_install_package(${PROJECT_NAME})

# =============================================================================
# Python Node Executables
# =============================================================================
# Install Python scripts as executable nodes
# These will be available via: ros2 run qcar2_laneseg_acc <node_name>

install(PROGRAMS
  qcar2_laneseg_acc/camera_buffer.py
  qcar2_laneseg_acc/tensor_fix.py
  qcar2_laneseg_acc/image_capture_node.py
  qcar2_laneseg_acc/tensorrt_inference_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# =============================================================================
# C++ Targets (Add your C++ nodes here when needed)
# =============================================================================
# Example for future C++ nodes:
# add_executable(my_cpp_node src/my_cpp_node.cpp)
# ament_target_dependencies(my_cpp_node rclcpp std_msgs sensor_msgs)
# install(TARGETS my_cpp_node DESTINATION lib/${PROJECT_NAME})

# =============================================================================
# Install launch, config, and model files
# =============================================================================
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
