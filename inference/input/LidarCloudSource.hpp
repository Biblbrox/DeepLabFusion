#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "PointCloudReader.hpp"
#include "core/base.hpp"

BEGIN_NAMESPACE

class LidarCloudSource : public PointCloudReader {
public:
  std::pair<bool, sensor_msgs::msg::PointCloud2> readCloud() override;
};

END_NAMESPACE