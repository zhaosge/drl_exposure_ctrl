cmake_minimum_required(VERSION 3.0.2)
project(drl_exposure_ctrl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 安装Python脚本
catkin_install_python(PROGRAMS
  infer.py
  train.py
  agent.py
  env.py
  log.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)