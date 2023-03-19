cmake_minimum_required(VERSION 3.8)
project(point_cloud_reader)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
#find_package(Boost REQUIRED COMPONENTS date_time)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)


FILE(GLOB CPP "*.cpp")

FILE(GLOB HPP "*.hpp" "*.h")
add_executable(${PROJECT_NAME} ${CPP} ${HPP})
ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs tf2 tf2_ros geometry_msgs laser_geometry)
target_include_directories(${PROJECT_NAME} PRIVATE "../")
#target_include_directories(${PROJECT_NAME} PRIVATE Boost::date_time)

install(TARGETS
        ${PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME}
        )

# Add PCL ##############################################################################################################
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})

target_link_directories(${PROJECT_NAME} PRIVATE ${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
