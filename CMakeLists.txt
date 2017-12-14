cmake_minimum_required(VERSION 3.7)
project(Clion_code)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED)

set(SOURCE_FILES stereo2_ex12_3.cpp)
#set(SOURCE_FILES image_capture.cpp)
add_executable(Clion_code ${SOURCE_FILES})

target_link_libraries( Clion_code ${OpenCV_LIBS})