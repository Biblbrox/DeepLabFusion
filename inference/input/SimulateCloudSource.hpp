#pragma once

#include <pcl/PCLPointCloud2.h>
#include <rclcpp/node.hpp>
#include <rclcpp/subscription.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "PointCloudReader.hpp"

BEGIN_NAMESPACE

class SimulateCloudSource : public PointCloudReader {
public:
  explicit SimulateCloudSource();
  std::pair<bool, sensor_msgs::msg::PointCloud2> readCloud() override;

private:
  void makeCloudAvailable(const sensor_msgs::msg::PointCloud2 & laserScan, const rclcpp::MessageInfo &info);

  std::shared_ptr<rclcpp::Node> m_simulationReaderNode;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr m_subscription;
  std::unique_lock<std::mutex> m_messageLock;
};

END_NAMESPACE