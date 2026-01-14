cmake_minimum_required(VERSION 3.8)
project(qcar2_laneseg_acc)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables (scripts)
install(PROGRAMS
  qcar2_laneseg_acc/segformer_decoder_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  qcar2_laneseg_acc/camera_buffer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  qcar2_laneseg_acc/tensor_fix.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install model files (.plan engines)
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
