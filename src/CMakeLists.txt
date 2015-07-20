SET(SRCS
        main.cpp
        mapstitch.cpp
)

SET(ROS_COMPONENT
        rosmain.cpp
        mapstitch.cpp
)

rosbuild_add_executable(mapstitch ${SRCS})
rosbuild_add_executable(ros_mapstitch ${ROS_COMPONENT})
