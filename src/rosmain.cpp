/** Copyright (c) 2013, TU Darmstadt, Philipp M. Scholl
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.  Redistributions in binary
form must reproduce the above copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other materials provided with
the distribution.  Neither the name of the <ORGANIZATION> nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.  THIS SOFTWARE IS PROVIDED
BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "nav_msgs/OccupancyGrid.h"
#include "mapstitch/mapstitch.h"
#include <tf/transform_broadcaster.h>
#include <opencv/highgui.h>
#include <unistd.h>
#include <string>

using namespace cv;
using namespace tf;
using namespace ros;
using namespace nav_msgs;

std::string save_stitch("");
double max_distance = 5.;
bool   debug = false;

struct stitch_maps {
  Mat   asimage;
  float resolution,
        origin_x,
        origin_y;
  char* frame_id;
} old_world, old_map;

Transform ourtf;
bool tf_valid = false;

Mat toMat(const nav_msgs::OccupancyGrid *map)
{
  uint8_t *data = (uint8_t*) map->data.data(),
           testpoint = data[0];
  bool mapHasPoints = false;

  Mat im(map->info.height, map->info.width, CV_8UC1);

  // transform the map in the same way the map_saver component does
  for (size_t i=0; i<map->data.size(); i++)
  {
    if (data[i] == 0)        im.data[i] = 254;
    else if (data[i] == 100) im.data[i] = 0;
    else im.data[i] = 205;

    // just check if there is actually something in the map
    if (i!=0) mapHasPoints = mapHasPoints || (data[i] != testpoint);
    testpoint = data[i];
  }

  // sanity check
  if (!mapHasPoints) { ROS_WARN("map is empty, ignoring update."); }

  return im;
}

void publish_stitch()
{
  static tf::TransformBroadcaster br;

  if (tf_valid)
    br.sendTransform(
      StampedTransform(ourtf,Time::now()+Duration(3.), old_world.frame_id, old_map.frame_id));
}

void update_tf(struct stitch_maps *w, struct stitch_maps *m)
{
  if (w == NULL) { ROS_INFO("world map not loaded"); return; }
  if (m == NULL) { ROS_INFO("map not loaded"); return; }

  // sanity check on input
  if (w->resolution != m->resolution) {
    ROS_WARN("map (%.3f) resolution differs from world resolution(%.3f)"
             "  and scaling not implemented, ignoring update",
             m->resolution, w->resolution);
    return;
  }

  if (debug) { // write images if debug
    imwrite("current_map.pgm", m->asimage);
    imwrite("current_world.pgm", w->asimage);
  }

  StitchedMap c(w->asimage,m->asimage, max_distance);

  // sanity checks
  if ((c.rot_deg == 0. && (int) c.transx == 0 && (int) c.transy == 0) ||
      (int) c.transx == INT_MAX || (int) c.transx == INT_MIN ||
      (int) c.transy == INT_MAX || (int) c.transy == INT_MIN)
  {
    ROS_INFO("homography estimation didn't work, not stitching.");
    return;
  }

  if (debug) { // write images if debug
    imwrite("current_stitch.pgm", c.get_stitch());
  }

  if (save_stitch.size() > 0) {
    imwrite(save_stitch.c_str(), c.get_stitch());
  }

  // publish this as the transformation between /world -> /map
  // The point-of-reference for opencv is the edge of the image, for ROS
  // this is the centerpoint of the image, which is why we translate each
  // point to the edge, apply rotation+translation from opencv and move
  // back to the center.
  float res = m->resolution;
  Mat H  = c.H;
  Mat E  = (Mat_<double>(3,3) << 1, 0,  m->origin_x,
                                 0, 1,  m->origin_y,
                                 0, 0, 1),
      E_ = (Mat_<double>(3,3) << 1, 0, -m->origin_x,
                                 0, 1, -m->origin_y,
                                 0, 0, 1);
  H.resize(3);
  H.at<double>(2,2) = 1.;
  H.at<double>(2,0) = H.at<double>(2,1) = 0.;
  H.at<double>(1,2) *= res;
  H.at<double>(0,2) *= res;
  H = E*H*E_;

  double rot = -1 *atan2(H.at<double>(0,1),H.at<double>(1,1)),
      transx = H.at<double>(0,2),
      transy = H.at<double>(1,2);

  ROS_INFO("stichted map with rotation %.5f radians and (%f,%f) translation",
      rot,transx,transy);

  ourtf = Transform( Quaternion(Vector3(0,0,1), rot),
                     Vector3(transx,transy,0) );
  tf_valid = true;

  if (m->frame_id  == w->frame_id)
    ROS_WARN("frame_id for world and map are the same, this will probably not work"
             "If map_server publishes your maps you might want to use _frame_id:=/world");
}

void update_stitch(struct stitch_maps *old_map,
                   const nav_msgs::OccupancyGrid& new_map)
{
  old_map->asimage    = toMat(&new_map);
  old_map->resolution = new_map.info.resolution;
  old_map->origin_x   = new_map.info.origin.position.x;
  old_map->origin_y   = new_map.info.origin.position.y;
  if (old_map->frame_id == NULL)
    old_map->frame_id   = strdup(new_map.header.frame_id.c_str());
  else if (strcmp(old_map->frame_id, new_map.header.frame_id.c_str())!=0) {
    free(old_map->frame_id);
    old_map->frame_id  = strdup(new_map.header.frame_id.c_str());
  }
}

void worldCallback(const nav_msgs::OccupancyGrid& new_world)
{
  update_stitch(&old_world, new_world);
  update_tf(&old_world, &old_map);
  publish_stitch();
}

void mapCallback(const nav_msgs::OccupancyGrid& new_map)
{
  publish_stitch();
}

void alignCallback(const nav_msgs::OccupancyGrid& new_map)
{
  update_stitch(&old_map, new_map);
  update_tf(&old_world, &old_map);
  publish_stitch();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_stitcher");
  ros::NodeHandle n;
  ros::NodeHandle param("~");

  // load the parameters
  param.getParam("save_stitch", save_stitch);
  param.getParam("max_distance", max_distance);
  param.getParam("debug", debug);

  // make sure these are initialized so we don't get a memory hole
  old_world.frame_id = old_map.frame_id = NULL;

  ros::Subscriber submap   = n.subscribe("map", 1000, mapCallback),
                  subworld = n.subscribe("world", 1000, worldCallback),
                  subalign = n.subscribe("align", 1000, alignCallback);
  ros::spin();

  return 0;
}
