#include <sensor_msgs/LaserScan.h>

#include "PointCloudReaderNode.hpp"

// using namespace rclcpp;

NAMESPACE::PointCloudReaderNode::PointCloudReaderNode()
    : Node("pointcloud_reader") {
  // auto default_qos = rclcpp::QoS(rclcpp::SystemDefaultsQoS());
  auto s = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", 10,
      std::bind(&PointCloudReaderNode::topicCallback, this,
                std::placeholders::_1, std::placeholders::_2));
}

void NAMESPACE::PointCloudReaderNode::topicCallback(
    const sensor_msgs::msg::LaserScan::ConstPtr &scan,
    const rclcpp::MessageInfo &info) const {
  for (const auto &i : scan->intensities)
    RCLCPP_INFO(this->get_logger(), "I heard: '%f'", i);
}
