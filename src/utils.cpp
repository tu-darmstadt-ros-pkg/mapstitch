#include "mapstitch/utils.h"

#include <ros/ros.h>

cv::Mat occupancyGridToCvMat(const nav_msgs::OccupancyGrid *map)
{
  uint8_t *data = (uint8_t*) map->data.data(),
           testpoint = data[0];
  bool mapHasPoints = false;

  cv::Mat im(map->info.height, map->info.width, CV_8UC1);

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

nav_msgs::OccupancyGrid cvMatToOccupancyGrid(const cv::Mat * im)
{
    nav_msgs::OccupancyGrid map;

    map.info.height = im->rows;
    map.info.width = im->cols;

    map.data.resize(map.info.width * map.info.height);

    for(size_t i = 0; i < map.data.size(); ++i)
    {
//        double map_val = (255. - im->data[i]) / 255.;
//        if(map_val > .65) map.data[i] = 100;
//        else if(map_val <= 0.196) map.data[i] = 0;
//        else map.data[i] = -1;

        uint8_t map_val = im->data[i];
        if(map_val == 0) map.data[i] = 100;
        else if(map_val == 254) map.data[i] = 0;
        else map.data[i] = -1;
    }

    return map;
}
