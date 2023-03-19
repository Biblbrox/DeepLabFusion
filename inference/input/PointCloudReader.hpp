#pragma once

#include "core/base.hpp"

BEGIN_NAMESPACE

class PointCloudReader {
public:
  virtual ~PointCloudReader() = default;
  virtual std::pair<bool, sensor_msgs::msg::PointCloud2> readCloud() = 0;
};

END_NAMESPACE
