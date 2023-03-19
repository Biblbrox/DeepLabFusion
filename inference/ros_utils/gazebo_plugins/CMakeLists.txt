cmake_minimum_required(VERSION 2.8.3)
project(gazebo_rosett_lidar_plugin)

######################################################################################################
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(laser_geometry REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_plugins REQUIRED)
find_package(gazebo_ros REQUIRED)

FILE(GLOB CPP "*.cpp")
FILE(GLOB HPP "*.hpp" "*.h")
add_library(${PROJECT_NAME} ${CPP} ${HPP})
ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs geometry_msgs laser_geometry RayPlugin)
target_include_directories(${PROJECT_NAME} PRIVATE "../")

install(TARGETS
        ${PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME}
        )

target_link_libraries(${PROJECT_NAME} RayPlugin)

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
