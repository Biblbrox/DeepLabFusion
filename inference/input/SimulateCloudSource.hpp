#pragma once

#include "PointCloudReader.hpp"

BEGIN_NAMESPACE

template <typename T>
class SimulateCloudSource : public lidarInference::PointCloudReader<T> {
public:
  pcl::PointCloud<T> readCloudFrame() override
  {

  }
};

END_NAMESPACE