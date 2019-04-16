cmake_minimum_required(VERSION 2.8.3)
project(ros_image_processing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  image_transport
  cv_bridge
  sensor_msgs
  std_msgs
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 3.2 REQUIRED)


include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


catkin_package()
include_directories(${catkin_INCLUDE_DIRS})


add_executable(TestAstra src/testAstra.cpp)
target_link_libraries(TestAstra ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(NodeImgProc src/main.cpp)
target_link_libraries(NodeImgProc ${catkin_LIBRARIES} ${OpenCV_LIBS} )


