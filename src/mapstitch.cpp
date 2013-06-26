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

StitchedMap::StitchedMap(Mat &img1, Mat &img2, float max_pairwise_distance)
    : is_valid(true)
{
  // load images, TODO: check that they're grayscale
  image1 = img1.clone();
  image2 = img2.clone();

  // create feature detector set.
  OrbFeatureDetector detector;
  OrbDescriptorExtractor dexc;
  BFMatcher dematc(NORM_HAMMING, false);

  // 1. extract keypoints
  detector.detect(image1, kpv1);
  detector.detect(image2, kpv2);

  // 2. extract descriptors
  dexc.compute(image1, kpv1, dscv1);
  dexc.compute(image2, kpv2, dscv2);

  // 3. match keypoints
  dematc.match(dscv1, dscv2, matches);

  // 4. find matching point pairs with same distance in both images
  for (size_t i=0; i<matches.size(); i++) {
    KeyPoint a1 = kpv1[matches[i].queryIdx],
             b1 = kpv2[matches[i].trainIdx];

    if (matches[i].distance > 30)
      continue;

    for (size_t j=0; j<matches.size(); j++) {
      KeyPoint a2 = kpv1[matches[j].queryIdx],
               b2 = kpv2[matches[j].trainIdx];

      if (matches[j].distance > 30)
        continue;

      if ( fabs(norm(a1.pt-a2.pt) - norm(b1.pt-b2.pt)) > max_pairwise_distance ||
           fabs(norm(a1.pt-a2.pt) - norm(b1.pt-b2.pt)) == 0)
        continue;

      matches_filtered.push_back(matches[j]);

      coord1.push_back(a1.pt);
      coord1.push_back(a2.pt);
      coord2.push_back(b1.pt);
      coord2.push_back(b2.pt);

      fil1.push_back(a1);
      fil1.push_back(a2);
      fil2.push_back(b1);
      fil2.push_back(b2);
    }
  }

  if (coord1.empty() || coord2.empty())
  {
    is_valid = false;
  }
  else
  {
      // 5. find homography
      H = estimateRigidTransform(coord2, coord1, false);

      if(H.empty() /*|| H.rows < 3 || H.cols < 3*/)
      {
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
  drawKeypoints(image1, kpv1, image1, Scalar(255,0,0));
  drawKeypoints(image2, kpv2, image2, Scalar(255,0,0));
  drawMatches(image1,fil1, image2,fil2, matches_filtered,out,Scalar::all(-1),Scalar::all(-1));
  std::cout << "total matches: " << matches.size() << " filtered matches: " << matches_filtered.size() << std::endl;
  return out;
}

Mat // return the stitched maps
StitchedMap::get_stitch()
{
  // create storage for new image and get transformations
  Mat warped_image(image2.size(), image2.type());
  warpAffine(image2,warped_image,H,warped_image.size(),INTER_NEAREST,BORDER_CONSTANT,205);

  Mat merged_image(min(image1.rows,warped_image.rows),min(image1.cols,warped_image.cols),warped_image.type());

  for(size_t i = 0; i < merged_image.size().area(); ++i)
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

StitchedMap::~StitchedMap() { }
