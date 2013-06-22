#ifndef MAPSTITCH_H
#define MAPSTITCH_H

#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>

using namespace cv;
using namespace std;

class StitchedMap
{
public:
  StitchedMap(Mat &im1, Mat &im2, float max_distance=5.);
  ~StitchedMap();

  Mat get_debug();
  Mat get_stitch();

  Mat H; // transformation matrix
  Mat image1, image2,
      dscv1, dscv2;
  bool is_valid;

  vector<KeyPoint> kpv1,kpv2;
  vector<KeyPoint> fil1,fil2;
  vector<Point2f>  coord1,coord2;
  vector<DMatch>   matches;
  vector<DMatch>   matches_filtered;

  double rot_deg,rot_rad,transx,transy,scalex,scaley;
};

#endif // MAPSTITCH_H
