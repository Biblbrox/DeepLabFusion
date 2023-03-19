#include <rclcpp/wait_set.hpp>

#include "SimulateCloudSource.hpp"

BEGIN_NAMESPACE

std::pair<bool, sensor_msgs::msg::PointCloud2>
lidarInference::SimulateCloudSource::readCloud() {
  rclcpp::WaitSet waitSet;
  waitSet.add_subscription(m_subscription);
  auto timeout = std::chrono::milliseconds(50);
  auto ret = waitSet.wait(timeout);
  if (ret.kind() == rclcpp::WaitResultKind::Ready) {
    sensor_msgs::msg::PointCloud2 laserScan;
    rclcpp::MessageInfo info;
    auto retTake = m_subscription->take(laserScan, info);
    if (retTake) {
      return {true, laserScan};
      // Received cloud
    } else {
      return {false, sensor_msgs::msg::PointCloud2()};
      // No message received
    }
  } else { // Timeout
    return {false, sensor_msgs::msg::PointCloud2()};
  }
}

SimulateCloudSource::SimulateCloudSource() {
  m_simulationReaderNode =
      std::make_shared<rclcpp::Node>("simulation_reader_node");
  m_subscription =
      m_simulationReaderNode
          ->create_subscription<sensor_msgs::msg::PointCloud2>(
              "/scan", 10,
              std::bind(&SimulateCloudSource::makeCloudAvailable, this,
                        std::placeholders::_1, std::placeholders::_2));
}

void SimulateCloudSource::makeCloudAvailable(
    const sensor_msgs::msg::PointCloud2 &laserScan,
    const rclcpp::MessageInfo &info) {
  std::cout << "Received cloud \n";
}

END_NAMESPACE
