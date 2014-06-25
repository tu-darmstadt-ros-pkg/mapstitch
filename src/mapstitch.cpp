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

  if (coord1.size() == 0){
    cout << "Point Vector is empty" << endl;
  }

  // 5. find homography
    H = estimateRigidTransform(coord2, coord1, false);
    
  // 6. calculate this stuff for information
  if (!H.empty()){
    rotation = 180.0 / M_PI * atan2(H.at<double>(0,1), H.at<double>(1,1)),
    transx = H.at<double>(0,2),
    transy = H.at<double>(1,2);

    scalex = sqrt(pow(H.at<double>(0,0),2) + pow(H.at<double>(0,1),2));
    scaley = sqrt(pow(H.at<double>(1,0),2) + pow(H.at<double>(1,1),2));
    
  }else{
    cout << "Error: Empty Transformation Matrix" << endl;
    exit(-1);
  }

}

Mat
StitchedMap::get_debug()
{
  Mat out;
  drawKeypoints(image1, kpv1, image1, Scalar(255,0,0));
  drawKeypoints(image2, kpv2, image2, Scalar(255,0,0));
  drawMatches(image1,fil1, image2,fil2, matches,out,Scalar::all(-1),Scalar::all(-1));
  return out;
}

Mat // return the stitched maps
StitchedMap::get_stitch()
{
  // create storage for new image and get transformations
  Mat image(image2.size(), image2.type());
  warpAffine(image2,image,H,image.size());

  // blend image1 onto the transformed image2
  addWeighted(image,.5,image1,.5,0.0,image);

  return image;
}

StitchedMap::~StitchedMap() { }
