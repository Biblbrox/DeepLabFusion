#include <pcl/PCLPointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/LaserScan.h>

#include "LidarCloudSource.hpp"
#include "PointCloudReaderNode.hpp"
#include "SimulateCloudSource.hpp"
#include "StereoCloudSource.hpp"

// using namespace rclcpp;

BEGIN_NAMESPACE

PointCloudReaderNode::PointCloudReaderNode(CloudSource sourceType)
    : Node("pointcloud_reader"), m_cloudSource(sourceType) {

  m_publisher = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "pointcloud_reader", 10);

  if (m_cloudSource == CloudSource::simulation_lidar) {
    m_cloudReader = std::make_shared<SimulateCloudSource>();
  } else if (m_cloudSource == CloudSource::stereo) {
    m_cloudReader = std::make_shared<StereoCloudSource>();
  } else if (m_cloudSource == CloudSource::lidar) {
    m_cloudReader = std::make_shared<LidarCloudSource>();
  }

  auto callbackGroup =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  this->create_wall_timer(
      std::chrono::milliseconds(100),
      [this]() {
        auto cloudMsg = m_cloudReader->readCloud();
        if (cloudMsg.first) {
          sensor_msgs::msg::PointCloud2 &cloud = cloudMsg.second;
          m_publisher->publish(cloud);
          cloudMsg = m_cloudReader->readCloud();
        }
      },
      callbackGroup);
}

END_NAMESPACE