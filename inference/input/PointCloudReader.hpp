#pragma once

#include <pcl/point_cloud.h>

#include "core/base.hpp"

BEGIN_NAMESPACE

/**
 * Real-time point coud reader
 */
 template <typename PointType>
class PointCloudReader {
public:
  virtual pcl::PointCloud<PointType> readCloudFrame() = 0;
};

END_NAMESPACE
