#ifndef MAPSTITCH_H
#define MAPSTITCH_H

#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>

#include <boost/shared_ptr.hpp>

using namespace cv;
using namespace std;

class StitchedMap
{
public:
  StitchedMap(Mat &im1, Mat &im2, float max_distance=5.);
  StitchedMap(Mat &img1, Mat &img2,cv::Mat &H_provided, float max_distance=5. );
  ~StitchedMap();

  Mat get_debug();
  Mat get_stitch();

  void printDebugOutput();
  bool isValid();

  Mat getTransformForThreePoints(const vector<DMatch>& matches,
                                 const vector<KeyPoint>& dest_q,
                                 const vector<KeyPoint>& input_t,
                                 const vector<int>& indices);

  bool isScaleValid(const cv::Mat& rigid_transform, double threshold_epsilon);

  Mat estimateHomographyRansac(const vector<DMatch>& matches,
                               const vector<KeyPoint>& dest_q,
                               const vector<KeyPoint>& input_t);

  bool pointSelectionPlausible(const std::vector<int>& indices,
                               const vector<DMatch>& matches,
                               const vector<KeyPoint>& dest_q,
                               const vector<KeyPoint>& input_t);

  bool transformPlausible(double origin_dist_threshold,
                          double resolution,
                          double origin_x,
                          double origin_y);



  double rot_deg,rot_rad,transx,transy,scalex,scaley;

  const Mat& getRigidTransform() const { return H; };

protected:

  Mat H; // transformation matrix

  Mat image1, image2,
      dscv1_q, dscv2_t;
  bool is_valid;

  vector<KeyPoint> kpv1_q,kpv2_t;

  std::vector<cv::Point2f> input_inliers;
  std::vector<cv::Point2f> dest_inliers;

  vector<DMatch>   matches;
  vector<DMatch>   matches_robust;
  vector<DMatch>   matches_filtered;

  boost::shared_ptr<OrbFeatureDetector> detector;
  boost::shared_ptr<OrbDescriptorExtractor> dexc;
  boost::shared_ptr<BFMatcher> dematc;
};

#endif // MAPSTITCH_H
