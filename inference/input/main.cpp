#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "PointCloudReaderNode.hpp"
#include "SimulateCloudSource.hpp"
#include "StereoCloudSource.hpp"

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NAMESPACE::PointCloudReaderNode>(
      NAMESPACE::CloudSource::simulation_lidar));
  rclcpp::shutdown();

  return 0;
}