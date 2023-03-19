#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "PointCloudReader.hpp"

BEGIN_NAMESPACE

class StereoCloudSource : public PointCloudReader {
public:
  explicit StereoCloudSource() {}
  std::pair<bool, sensor_msgs::msg::PointCloud2> readCloud();

private:
};

END_NAMESPACE