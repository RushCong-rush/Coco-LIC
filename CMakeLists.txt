cmake_minimum_required(VERSION 3.0.2)
project(cocolic)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  eigen_conversions
  cv_bridge
  roslib
  rosbag
  tf
  message_generation
  image_transport
  livox_ros_driver
)

# FIND_PACKAGE(Boost REQUIRED COMPONENTS thread)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem iostreams program_options system serialization thread)
if(Boost_FOUND)
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
endif()

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(opengv REQUIRED)
# ROS Noetic on Ubuntu 20.04 ships PCL 1.10. Coco-LIC only uses APIs
# available in that release.
find_package(PCL 1.10 REQUIRED)

# find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  feature_cloud.msg
  imu_array.msg
  pose_array.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs 
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS src
  CATKIN_DEPENDS std_msgs
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  src
  ${catkin_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  # ${Sophus_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  src/camera
  src/camera/loam/include
  src/camera/tools/
  src/camera/rgb_map
)
list(APPEND thirdparty_libraries
        ${YAML_CPP_LIBRARIES}
        ${Boost_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        opengv
        ${CERES_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SERIALIZATION_LIBRARY} # serialization
)

add_library(spline_lib src/spline/trajectory.cpp)
add_dependencies(spline_lib ${catkin_EXPORTED_TARGETS})
target_link_libraries(spline_lib ${thirdparty_libraries})

file(GLOB lidar_odometry_files
    "src/lidar/lidar_handler.cpp"
    "src/lidar/velodyne_feature_extraction.cpp",
    "src/lidar/livox_feature_extraction.cpp"
)
add_library(lidar_lib ${lidar_odometry_files})
add_dependencies(lidar_lib ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_lib spline_lib ${thirdparty_libraries})

file(GLOB r3live_files
    "src/camera/loam/*.cpp"
    "src/camera/optical_flow/*.cpp"
    "src/camera/rgb_map/*.cpp"
    "src/camera/*.cpp"
)
add_library(r3live_lib ${r3live_files})
add_dependencies(r3live_lib ${catkin_EXPORTED_TARGETS})
target_link_libraries(r3live_lib ${thirdparty_libraries})

add_executable(odometry_node 
   src/odometry_node.cpp
   src/imu/imu_state_estimator.cpp
   src/imu/imu_initializer.cpp
   src/odom/msg_manager.cpp
   src/odom/trajectory_manager.cpp
   src/odom/trajectory_estimator.cpp
   src/odom/odometry_manager.cpp
   src/odom/factor/analytic_diff/marginalization_factor.cpp
   src/odom/factor/analytic_diff/wnoj_rotation_factor.cpp
   src/utils/parameter_struct.cpp
)
add_dependencies(odometry_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)

target_link_libraries(odometry_node 
   spline_lib 
   lidar_lib 
   r3live_lib
   ${thirdparty_libraries}
  #  fmt::fmt
)

if(CATKIN_ENABLE_TESTING)
  add_executable(empty_lidar_window_test
                 tests/test_empty_lidar_window.cpp
                 src/odom/trajectory_manager.cpp
                 src/imu/imu_state_estimator.cpp
                 src/odom/trajectory_estimator.cpp
                 src/odom/factor/analytic_diff/marginalization_factor.cpp
                 src/odom/factor/analytic_diff/wnoj_rotation_factor.cpp
                 src/utils/parameter_struct.cpp)
  target_link_libraries(empty_lidar_window_test
                        spline_lib r3live_lib ${thirdparty_libraries})
  add_test(NAME empty_lidar_window_test COMMAND empty_lidar_window_test
           ${CMAKE_CURRENT_SOURCE_DIR}/config)
  set_tests_properties(empty_lidar_window_test PROPERTIES
                      FAIL_REGULAR_EXPRESSION "GetIdxT wrong")
  add_executable(marginalization_manifold_test
                 tests/test_marginalization_manifold.cpp
                 src/odom/trajectory_estimator.cpp
                 src/odom/factor/analytic_diff/marginalization_factor.cpp
                 src/odom/factor/analytic_diff/wnoj_rotation_factor.cpp
                 src/utils/parameter_struct.cpp)
  target_link_libraries(marginalization_manifold_test
                        spline_lib r3live_lib ${thirdparty_libraries})
  add_test(NAME marginalization_manifold_test COMMAND marginalization_manifold_test)
  add_executable(erp_ransac_test tests/test_erp_ransac.cpp)
  target_link_libraries(erp_ransac_test r3live_lib ${thirdparty_libraries})
  add_test(NAME erp_ransac_test COMMAND erp_ransac_test)
  add_executable(erp_periodic_tracking_test tests/test_erp_periodic_tracking.cpp)
  target_link_libraries(erp_periodic_tracking_test r3live_lib ${thirdparty_libraries})
  add_test(NAME erp_periodic_tracking_test COMMAND erp_periodic_tracking_test)
  add_executable(camera_geometry_test tests/test_camera_geometry.cpp)
  target_link_libraries(camera_geometry_test ${catkin_LIBRARIES})
  add_test(NAME camera_geometry_test COMMAND camera_geometry_test)
  add_executable(erp_visual_residual_test tests/test_erp_visual_residual.cpp)
  target_link_libraries(erp_visual_residual_test ${catkin_LIBRARIES})
  add_test(NAME erp_visual_residual_test COMMAND erp_visual_residual_test)
  add_executable(robust_wnoa_process_factor_test
                 tests/test_robust_wnoa_process_factor.cpp)
  target_link_libraries(robust_wnoa_process_factor_test
                        ${thirdparty_libraries})
  add_test(NAME robust_wnoa_process_factor_test
           COMMAND robust_wnoa_process_factor_test)
  add_executable(wnoj_translation_factor_test tests/test_wnoj_translation_factor.cpp)
  target_link_libraries(wnoj_translation_factor_test ${thirdparty_libraries})
  add_test(NAME wnoj_translation_factor_test COMMAND wnoj_translation_factor_test)
  add_executable(wnoj_rotation_factor_test tests/test_wnoj_rotation_factor.cpp
                 src/odom/factor/analytic_diff/wnoj_rotation_factor.cpp)
  target_link_libraries(wnoj_rotation_factor_test ${thirdparty_libraries})
  add_test(NAME wnoj_rotation_factor_test COMMAND wnoj_rotation_factor_test)
endif()
