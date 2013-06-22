SET(HDRS
        ../include/mapstitch/mapstitch.h
)


SET(SRCS
        main.cpp
        mapstitch.cpp
)

SET(ROS_COMPONENT
        rosmain.cpp
        mapstitch.cpp
)

rosbuild_add_executable(mapstitch_tool ${SRCS} ${HDRS})
rosbuild_add_executable(ros_mapstitch ${ROS_COMPONENT} ${HDRS})

rosbuild_add_library(${PROJECT_NAME} mapstitch.cpp utils.cpp)
