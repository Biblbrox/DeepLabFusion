#pragma once

#include <laser_geometry/laser_geometry.hpp>
#include <message_filters/subscriber.h>
#include <rclcpp/rclcpp.hpp>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include "core/base.hpp"

BEGIN_NAMESPACE

class PointCloudReaderNode : public rclcpp::Node {
public:
  PointCloudReaderNode();

private:
  void topicCallback(const sensor_msgs::msg::LaserScan::ConstPtr &scan,
                     const rclcpp::MessageInfo &info) const;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_subscription;
};

END_NAMESPACE
