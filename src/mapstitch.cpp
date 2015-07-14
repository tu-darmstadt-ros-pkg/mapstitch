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
#include "mapstitch/mapstitch.h"
#include "math.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/eigen.hpp>

#include <robust_matcher/robust_matcher.h>

StitchedMap::StitchedMap(Mat &img1, Mat &img2, float max_pairwise_distance)
    : is_valid(true)
{
  if (img1.empty() ){
    is_valid = false;
    std::cout << "img1 is empty, aborting.";
    return;
  }

  if (img2.empty() ){
    is_valid = false;
    std::cout << "img2 is empty, aborting.";
    return;
  }

  // load images, TODO: check that they're grayscale
  image1 = img1.clone();
  image2 = img2.clone();

  int nfeatures = 500;
  float scaleFactor = 1.2f;
  int nlevels=8;
  int edgeThreshold = 31;
  int firstLevel = 0;
  int WTA_K=2;
  int scoreType=ORB::HARRIS_SCORE;
  int patchSize=31;

  // create feature detector set.
  OrbFeatureDetector* detector = new OrbFeatureDetector(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
  OrbDescriptorExtractor* dexc = new OrbDescriptorExtractor(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);
  BFMatcher* dematc = new BFMatcher(NORM_HAMMING, false);

  /*
  RobustMatcher robust_matcher;

  robust_matcher.setFeatureDetector(detector);
  robust_matcher.setDescriptorExtractor(dexc);
  robust_matcher.setDescriptorMatcher(dematc);

  //vector<KeyPoint> keypoints_image2;
  robust_matcher.computeKeyPoints(image2, keypoints_image2);

  Mat descriptors_image2;
  robust_matcher.computeDescriptors(image2, keypoints_image2, descriptors_image2);

  robust_matcher.robustMatch(image1, matches_robust, keypoints_image1, descriptors_image2);

  std::cout << "Robust matches: " << matches_robust.size()<< "\n";
  */

  // 1. extract keypoints
  detector->detect(image1, kpv1);
  detector->detect(image2, kpv2);

  // 2. extract descriptors
  dexc->compute(image1, kpv1, dscv1);
  dexc->compute(image2, kpv2, dscv2);

  // 3. match keypoints
  dematc->match(dscv1, dscv2, matches);

  std::cout << "kpv1 size: " << kpv1.size() << " kpv2 size: " << kpv2.size() << "\n";


  size_t idx = 0;

  float min_dist = std::numeric_limits<float>::max();
  int min_index = -1;

  int good_count = 0;

  // 4. find matching point pairs with same distance in both images
  for (size_t i=0; i<matches.size(); i++) {
    KeyPoint a1 = kpv1[matches[i].queryIdx],
             b1 = kpv2[matches[i].trainIdx];

    if (matches[i].distance > 30)
      continue;

    for (size_t j=0; j<matches.size(); j++) {

      if (i == j)
        continue;

      KeyPoint a2 = kpv1[matches[j].queryIdx],
               b2 = kpv2[matches[j].trainIdx];

      if (matches[j].distance > 30)
        continue;

      float dist = fabs(norm(a1.pt-a2.pt) - norm(b1.pt-b2.pt));

      if ( dist < max_pairwise_distance){
        good_count++;
        if (dist < min_dist){
          min_dist = dist;
          min_index = j;
        }
      }
    }

    if(good_count > 0){

      matches_filtered.push_back(matches[min_index]);
      matches_filtered.back().queryIdx = idx;
      matches_filtered.back().trainIdx = idx;

      coord1.push_back(a1.pt);
      //coord1.push_back(a2.pt);
      coord2.push_back(b1.pt);
      //coord2.push_back(b2.pt);

      fil1.push_back(a1);
      fil2.push_back(b1);

      //std::cout << "mf: " << matches_filtered.back().queryIdx << " " << matches_filtered.back().trainIdx << "\n";
      //std::cout << "a1: " << a1.pt.x << " " << a1.pt.y << "\n";
      //std::cout << "b1: " << b1.pt.x << " " << b1.pt.y << "\n";
      //std::cout << "a2: " << a2.pt.x << " " << a2.pt.y << "\n";
      //std::cout << "b2: " << b2.pt.x << " " << b2.pt.y << "\n";

      ++idx;
    }

  }

  std::cout << "num filtered matches: " << matches_filtered.size() << "\n";

  if (coord1.empty() || coord2.empty())
  {
    is_valid = false;
  }
  else
  {
    H = this->estimateHomographyRansac(matches_filtered, fil1, fil2);
    //H = this->estimateHomographyRansac(matches, kpv1, kpv2);
    //H = this->estimateHomographyRansac(matches_robust, keypoints_image2, keypoints_image1);


    if(H.empty() /*|| H.rows < 3 || H.cols < 3*/)
    {
      std::cout << "H Matrix empty\n";
      is_valid = false;
    }
    else
    {
      // 6. calculate this stuff for information
      rot_rad  = atan2(H.at<double>(0,1),H.at<double>(1,1));
      rot_deg  = 180./M_PI* rot_rad;
      transx   = H.at<double>(0,2);
      transy   = H.at<double>(1,2);
      scalex   = sqrt(pow(H.at<double>(0,0),2)+pow(H.at<double>(0,1),2));
      scaley   = sqrt(pow(H.at<double>(1,0),2)+pow(H.at<double>(1,1),2));
    }

  }
}

Mat
StitchedMap::get_debug()
{
  Mat out;
  std::cout << "total matches: " << matches.size() << " filtered matches: " << matches_filtered.size() << std::endl;
  std::cout << "fil1 size: " << fil1.size() << " fil2 size: " << fil2.size() << "\n";
  drawKeypoints(image1, kpv1, image1, Scalar(255,0,0));
  drawKeypoints(image2, kpv2, image2, Scalar(255,0,0));

  for (size_t i = 0; i <coord1.size();++i){
    cv::circle(image1, coord1[i], 7, cv::Scalar(0,0 ,255));
    cv::circle(image2, coord2[i], 7, cv::Scalar(0,0 ,255));
  }

  for (size_t i = 0; i <input_inliers.size();++i){
    cv::circle(image1, dest_inliers[i], 9, cv::Scalar(255, 0 ,255), 2);
    cv::circle(image2, input_inliers[i], 9, cv::Scalar(255, 0 ,255), 2);
  }


  if (this->is_valid){
    drawMatches(image1,fil1, image2,fil2, matches_filtered,out,Scalar::all(-1),Scalar::all(-1));
  }else{
    /*
    Mat img_matches = Mat(image1.cols+image2.cols,image1.rows,image1.type());//set size as combination of img1 and img2


    Mat left(img_matches, Rect(0, 0, image1.cols, image1.rows)); // Copy constructor
    image1.copyTo(left);
    Mat right(img_matches, Rect(image1.cols, 0, image1.cols, image1.rows)); // Copy constructor
    image2.copyTo(right);
    */

    out = image1;
  }
  return out;
}

Mat // return the stitched maps
StitchedMap::get_stitch()
{
  if (!is_valid){
    std::cout << "Trying to get stitch despite not being valid, returning empty Mat.\n";
    Mat empty;
    return empty;
  }

  // create storage for new image and get transformations
  Mat warped_image(image2.size(), image2.type());
  warpAffine(image2,warped_image,H,warped_image.size(),INTER_NEAREST,BORDER_CONSTANT,205);

  Mat merged_image(min(image1.rows,warped_image.rows),min(image1.cols,warped_image.cols),warped_image.type());

  int area = merged_image.size().area();

  for(int i = 0; i < area; ++i)
  {
      // if cell is free in both maps
      if(image1.data[i] > 230 && warped_image.data[i] > 230)
      {
          merged_image.data[i] = 254;
      }
      // if cell is occupied in either map
      else if(image1.data[i] < 10 || warped_image.data[i] < 10)
      {
          merged_image.data[i] = 0;
      }
      // if cell is unknown in one and known in the other map
      else if(image1.data[i] > 200 && image1.data[i] < 210
              && (warped_image.data[i] < 200 || warped_image.data[i] > 210))
      {
          merged_image.data[i] = warped_image.data[i];
      }
      else if(warped_image.data[i] > 200 && warped_image.data[i] < 210
              && (image1.data[i] < 200 || image1.data[i] > 210))
      {
          merged_image.data[i] = image1.data[i];
      }
      // else the cell is unknown
      else
      {
          merged_image.data[i] = 205;
      }
  }

  // blend image1 onto the transformed image2
  //addWeighted(warped_image,.5,image1,.5,0.0,warped_image);

  return merged_image;
}

void StitchedMap::printDebugOutput()
{
  cout << "rotation: "          << rot_deg << endl
       << "translation (x,y): " << transx << ", " << transy << endl
       << "matrix: "            << H << endl;
}

bool StitchedMap::isValid()
{
  return is_valid;
}


Mat StitchedMap::getTransformForThreePoints(const vector<DMatch>& matches,
                               const vector<KeyPoint>& dest,
                               const vector<KeyPoint>& input,
                               const vector<int>& indices)
{
  std::vector<cv::Point2f> input_ransac;
  std::vector<cv::Point2f> dest_ransac;

  for(int i = 0; i < 3;++i){
    input_ransac.push_back(input[matches[indices[i]].queryIdx].pt);
    dest_ransac.push_back(dest[matches[indices[i]].trainIdx].pt);
  }

  return estimateRigidTransform(input_ransac, dest_ransac, false);
}

Mat StitchedMap::estimateHomographyRansac(const vector<DMatch>& matches,
                                          const vector<KeyPoint>& dest,
                                          const vector<KeyPoint>& input)
{

  float inlier_dist_threshold = 8.0;

  float inlier_dist_threshold_squared = inlier_dist_threshold * inlier_dist_threshold;

  CvRNG rng( 0xFFFFFFFF );

  std::vector<int> idx;
  idx.resize(3);

  Mat rigid_transform;

  int num_iterations = 0;

  int max_num_inliers = -1;
  std::vector<int> best_indices_vector;

  std::vector<Eigen::Vector2f> input_eigen;
  input_eigen.resize(input.size());
  for (size_t i = 0; i < input.size(); ++i){
    input_eigen[i] = Eigen::Vector2f(input[i].pt.x, input[i].pt.y);
  }

  std::vector<Eigen::Vector2f> dest_eigen;
  dest_eigen.resize(dest.size());
  for (size_t i = 0; i < dest.size(); ++i){
    dest_eigen[i] = Eigen::Vector2f(dest[i].pt.x, dest[i].pt.y);
  }



  while (num_iterations < 500){

    for(int i = 0; i < 3;++i){
      idx[i] = cvRandInt(&rng) % matches.size();
    }

    rigid_transform = getTransformForThreePoints(matches,
                                                 dest,
                                                 input,
                                                 idx);

    if (!rigid_transform.empty()){

      Eigen::AffineCompact2f transform;

      cv2eigen(rigid_transform, transform.matrix());

      int num_inliers = 0;
      for (size_t j = 0; j < matches.size(); ++j){

        Eigen::Vector2f transformed_to_dest = transform * input_eigen[matches[j].queryIdx];

        float dist_squared = (transformed_to_dest - dest_eigen[matches[j].trainIdx]).squaredNorm();

        if (dist_squared < inlier_dist_threshold_squared ){
          ++num_inliers;
        }
      }

      if (num_inliers > max_num_inliers){
        max_num_inliers = num_inliers;
        best_indices_vector = idx;

        std::cout << "inlier number increased to: " << max_num_inliers << " during ransac\n";
      }
    }

    ++num_iterations;
  }

  rigid_transform = getTransformForThreePoints(matches,
                                               dest,
                                               input,
                                               best_indices_vector);

  Eigen::AffineCompact2f transform;

  cv2eigen(rigid_transform, transform.matrix());

  int num_inliers = 0;
  for (size_t j = 0; j < matches.size(); ++j){

    Eigen::Vector2f transformed_to_dest = transform * input_eigen[matches[j].queryIdx];

    float dist_squared = (transformed_to_dest - dest_eigen[matches[j].trainIdx]).squaredNorm();

    if (dist_squared < inlier_dist_threshold_squared){
      ++num_inliers;
      input_inliers.push_back(input[matches[j].queryIdx].pt);
      dest_inliers.push_back(dest[matches[j].trainIdx].pt);

    }
  }

  std::cout << "Num inliers for best iter: " << num_inliers << "\n";

  Mat best_estimate_transform = estimateRigidTransform(input_inliers, dest_inliers, false);

  if (!best_estimate_transform.empty()){
    rigid_transform = best_estimate_transform;
    std::cout << "Inlier rigid transform estimation successful.\n";
  }else{
    std::cout << "Inlier rigid transform estimation failed, using prior best estimate\n";
  }

  return rigid_transform;
}


StitchedMap::~StitchedMap() { }
