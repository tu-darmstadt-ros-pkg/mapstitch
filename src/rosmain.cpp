#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "mapstitch/mapstitch.h"
#include <tf/transform_broadcaster.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace tf;
using namespace ros;
using namespace nav_msgs;

double max_distance = 5.;
bool   debug = false;
nav_msgs::OccupancyGrid old_world,
                        old_map;

Mat toMat(const nav_msgs::OccupancyGrid *map)
{
  uint8_t *data = (uint8_t*) map->data.data(),
           testpoint = data[0];
  bool mapHasPoints = false;

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

  // sanity check
  if (!mapHasPoints) {
    ROS_INFO("map is empty, ignoring update."); 
  }
  return im;
}

void publish_stitch(const nav_msgs::OccupancyGrid* w,
                    const nav_msgs::OccupancyGrid* m)
{
  static tf::TransformBroadcaster br;

  if (w == NULL) { ROS_INFO("world map not loaded"); return; }
  if (m == NULL)   { ROS_INFO("map not loaded"); return; }

  // sanity check on input
  if (w->info.resolution != m->info.resolution) {
    ROS_WARN("map (%.3f) resolution differs from world resolution(%.3f)"
             "  and scaling not implemented, ignoring update",
             m->info.resolution, w->info.resolution);
    return;
  }

  Mat imw = toMat(w), imm = toMat(m);
  StitchedMap c(imw,imm, max_distance);
 
  // sanity checks
  if ((c.rotation == 0. && (int) c.transx == 0 && (int) c.transy == 0) ||
      (int) c.transx == INT_MAX || (int) c.transx == INT_MIN ||
      (int) c.transy == INT_MAX || (int) c.transy == INT_MIN)
  {
    ROS_INFO("homography estimation didn't work, not stitching.");
    return;
  }

  if (debug) { // write images if debug
    imwrite("current_map.pgm", imm);
    imwrite("current_world.pgm", imw);
    imwrite("current_stitch.pgm", c.get_stitch());
  }

  // publish this as the transformation between /world -> /map
  // The point-of-reference for opencv is the edge of the image, for ROS
  // this is the centerpoint of the image, which is why we translate each
  // point to the edge, apply rotation+translation from opencv and move
  // back to the center.
  float res = m->info.resolution;
  Mat H  = c.H;
  Mat E  = (Mat_<double>(3,3) << 1, 0, m->info.origin.position.x,
                                 0, 1, m->info.origin.position.y,
                                 0, 0, 1),
      E_ = (Mat_<double>(3,3) << 1, 0, -m->info.origin.position.x,
                                 0, 1, -m->info.origin.position.y,
                                 0, 0, 1);
  H.resize(3);
  H.at<double>(2,2) = 1.;
  H.at<double>(2,0) = H.at<double>(2,1) = 0.;
  H.at<double>(1,2) *= res;
  H.at<double>(0,2) *= res;
  H = E*H*E_;

  double rot = atan2(H.at<double>(0,1),H.at<double>(1,1)),
      transx = -1 * H.at<double>(0,2),
      transy = -1 * H.at<double>(1,2);

  ROS_INFO("stichted map with rotation %.1f deg and (%f,%f) translation",
      rot,transx,transy);

  br.sendTransform( StampedTransform(
        Transform( Quaternion(Vector3(0,0,1), rot),
                   Vector3(transx,transy,0) ),
        Time::now(), m->header.frame_id, w->header.frame_id));

  if (m->header.frame_id  == w->header.frame_id)
    ROS_WARN("frame_id for world and map are equale, this will probably not work"
             "If map_server publishes your maps you might want to use _frame_id:=/world");
}


void worldCallback(const nav_msgs::OccupancyGrid& new_world)
{
  publish_stitch(&old_map, &new_world);
  old_world =  new_world;
}

void mapCallback(const nav_msgs::OccupancyGrid& new_map)
{
  publish_stitch(&new_map, &old_world);
  old_map = new_map;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_stitcher");
  ros::NodeHandle n;
  ros::NodeHandle param("~");

  // load the parameters
  param.getParam("max_distance", max_distance);
  param.getParam("debug", debug);

  ros::Subscriber submap   = n.subscribe("map", 1000, mapCallback),
                  subworld = n.subscribe("world", 1000, worldCallback);
  ros::spin();

  return 0;
}
