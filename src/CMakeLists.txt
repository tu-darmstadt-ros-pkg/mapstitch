SET(SRCS
        main.cpp
        mapstitch.cpp
)

#QT4_WRAP_CPP(MergingDialog_MOC_SRCS ${MergingDialog_MOC_HDRS})
#ADD_EXECUTABLE(mergingdialog ${MergingDialog_SRCS} ${MergingDialog_MOC_SRCS})
#TARGET_LINK_LIBRARIES(mergingdialog ${QT_LIBRARIES} ${OpenCV_LIBS})

#TARGET_LINK_LIBRARIES(mergingdialog ${QT_LIBRARIES} ${OpenCV_LIBS})
rosbuild_add_executable(mapstitch ${SRCS})
