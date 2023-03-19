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

#include "PointCloudReader.hpp"
#include "core/base.hpp"

BEGIN_NAMESPACE

enum class CloudSource { simulation_lidar, stereo, lidar };

class PointCloudReaderNode : public rclcpp::Node {
public:
  explicit PointCloudReaderNode(CloudSource sourceType);
private:
  std::shared_ptr<PointCloudReader> m_cloudReader;
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> m_publisher;
  CloudSource m_cloudSource;
};

END_NAMESPACE
