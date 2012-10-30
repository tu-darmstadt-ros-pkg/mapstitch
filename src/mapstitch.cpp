#include "mapstitch.h"
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
    Point2f a1 = kpv1[matches[i].queryIdx].pt,
            b1 = kpv2[matches[i].trainIdx].pt;

    for (size_t j=0; j<matches.size(); j++) {
      Point2f a2 = kpv1[matches[j].queryIdx].pt,
              b2 = kpv2[matches[j].trainIdx].pt;

      if ( fabs(norm(a1-a2) - norm(b1-b2)) > max_pairwise_distance)
        continue;

      coord1.push_back(a1);
      coord1.push_back(a2);
      coord2.push_back(b1);
      coord2.push_back(b2);
    }
  }

  // 5. find homography
  H = estimateRigidTransform(coord2, coord1, false);

  // 6. calculate this stuff for information
  rotation = 180./M_PI*atan2(H.at<double>(0,1),H.at<double>(1,1)),
  transx   = H.at<double>(0,2),
  transy   = H.at<double>(1,2);
  scalex   = sqrt(pow(H.at<double>(0,0),2)+pow(H.at<double>(0,1),2));
  scaley   = sqrt(pow(H.at<double>(1,0),2)+pow(H.at<double>(1,1),2));
}

Mat
StitchedMap::get_debug()
{
  Mat out;
  drawKeypoints(image1, kpv1, image1, Scalar(255,0,0));
  drawKeypoints(image2, kpv2, image2, Scalar(255,0,0));
  drawMatches(image1,kpv1, image2,kpv2, matches,out,Scalar::all(-1),Scalar::all(-1));
  return out;
}

Mat // return the stitched maps
StitchedMap::get_stitch()
{
  // calculate borders of transformed image2
  //int w = image2.size().width,
  //    h = image2.size().height;

  //Mat x[] = { H*(Mat_<double>(3,1) << 0,0,1),
  //            H*(Mat_<double>(3,1) << w,0,1),
  //            H*(Mat_<double>(3,1) << 0,h,1),
  //            H*(Mat_<double>(3,1) << w,h,1) };

  //double minx=x[0].at<double>(0,0),maxx=x[0].at<double>(0,0),
  //       miny=x[0].at<double>(1,0),maxy=x[0].at<double>(1,0);

  //for (size_t i=0; i<sizeof(x)/sizeof(*x); i++) {
  //  minx = min(minx, x[i].at<double>(0,0));
  //  maxx = max(maxx, x[i].at<double>(0,0));
  //  miny = min(miny, x[i].at<double>(1,0));
  //  maxy = max(maxy, x[i].at<double>(1,0));
  //}

  //Size s(fabs(minx)+fabs(maxx),
  //       fabs(miny)+fabs(maxy));

  //cout << s.width << " " << s.height << endl;

  // create storage for new image and get transformations
  Mat image(image2.size(), image2.type());
  warpAffine(image2,image,H,image.size());

  // blend image1 onto the transformed image2
  addWeighted(image,.5,image1,.5,0.0,image);

  return image;
}

StitchedMap::~StitchedMap() { }
