#pragma once

#include <gazebo-11/gazebo/common/Plugin.hh>
#include <gazebo-11/gazebo/plugins/RayPlugin.hh>
#include <gazebo-11/gazebo/sensors/RaySensor.hh>
#include <gazebo-11/gazebo/sensors/Sensor.hh>
#include <sdformat-9.7/sdf/Element.hh>
#include <sdformat-9.7/sdf/sdf.hh>


namespace gazebo {
class RosettLidarPlugin : public gazebo::RayPlugin {
public:
  explicit RosettLidarPlugin();

public:
  void Load(sensors::SensorPtr _parent, sdf::ElementPtr _sdf);
  ~RosettLidarPlugin();
};

} // namespace gazebo
