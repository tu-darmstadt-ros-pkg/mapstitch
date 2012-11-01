#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "mapstitch.h"
#include <tf/transform_broadcaster.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace tf;
using namespace ros;

double max_distance = 5.;
bool debug = false;
Mat world_map;
float world_resolution = 0.;
String map_frame = "map";

void mapCallback(const nav_msgs::OccupancyGridConstPtr& map)
{
  uint8_t *data = (uint8_t*) map->data.data(),
           testpoint = data[0];
  bool mapHasPoints = false;
  static tf::TransformBroadcaster br;
  float resolution = map->info.resolution;

  // check if resolutions are matching if possible
  if (world_resolution != 0. && world_resolution != resolution)
    ROS_WARN("resolution of world map (%.2f) does not match /map resolution"
        "(%.2f). This might fail!", world_resolution, resolution);

  Mat im(map->info.height, map->info.width, CV_8UC1, data);

  // transform the map in the same way the map_saver component does
  for (size_t i=0; i<map->data.size(); i++)
  {
    if (map->data[i] == 0)   data[i] = 254;
    else if (map->data[i] == 100) data[i] = 0;
    else data[i] = 205;

    // just check if there is actually something in the map
    if (i!=0) mapHasPoints = mapHasPoints || (data[i] != testpoint);
    testpoint = data[i];
  }

  // sanity checks
  if (!mapHasPoints) {
    ROS_INFO("map is empty");
    return;
  }

  // generate the stitch
  StitchedMap c(world_map,im, max_distance);

  // sanity checks
  if ((c.rotation == 0. && (int) c.transx == 0 && (int) c.transy == 0) ||
      (int) c.transx == INT_MAX || (int) c.transx == INT_MIN ||
      (int) c.transy == INT_MAX || (int) c.transy == INT_MIN)
  {
    ROS_INFO("homography estimation didn't work, not stitching.");
    return;
  }

  if (debug) { // write images if debug
    imwrite("current_map.pgm", im);
    imwrite("current_stitch.pgm", c.get_stitch());
  }

  ROS_INFO("stichted map with rotation %.1f deg and (%f,%f) translation",
      c.rotation, c.transx, c.transy);

  // publish this as the transformation between /world -> /map
  tf::Transform transform;
  transform.setOrigin( Vector3(c.transx*resolution,c.transy*resolution, 0.) );
  transform.setRotation( Quaternion(Vector3(0,0,1), c.rotation*M_PI/180.) );
  br.sendTransform(StampedTransform(transform, Time::now(), "world", map_frame));
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_stitcher");
  ros::NodeHandle n;
  ros::NodeHandle param("~");
  std::string world_map_file;

  // load the parameters
  param.getParam("max_distance", max_distance);
  param.getParam("debug", debug);
  param.getParam("map_frame", map_frame);

  // load the world map
  if (!param.getParam("world_map",world_map_file)) {
    ROS_FATAL("neccesary parameter 'world_map', the map to align to, is missing");
    exit(-1);
  }

  // 0 == grayscale
  // transposed because the map_saver component saves a mirrored map
  world_map = imread(world_map_file.c_str(), 0).t();

  if (world_map.data == NULL) {
    ROS_FATAL("unable to load '%s'", world_map_file.c_str());
    exit(-1);
  }

  ROS_WARN("resolution checking for world_map and /map is not implemented");

  ros::Subscriber sub = n.subscribe("map", 1000, mapCallback);
  ros::spin();

  return 0;
}
