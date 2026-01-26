"""Setup file for qcar2_laneseg_acc Python module.

This file is used by ament_cmake_python via ament_python_install_package()
in CMakeLists.txt. It configures the Python package for installation.

Note: In a hybrid ament_cmake + Python package, launch/config/models are
installed via CMakeLists.txt, not through setup.py data_files.
"""

from setuptools import find_packages, setup

package_name = 'qcar2_laneseg_acc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hackbrian',
    maintainer_email='hackbrian@example.com',
    description='Image-Processing semantic segmentation pipeline for QCar2 using Isaac ROS v2.1 NITROS',
    license='MIT',
    tests_require=['pytest'],
    # Entry points are NOT used in ament_cmake hybrid packages
    # The executables are installed directly via CMakeLists.txt install(PROGRAMS ...)
)
