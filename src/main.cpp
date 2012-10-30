#include <string>
#include <iostream>
#include <algorithm>
#include <opencv/highgui.h>
#include "tclap/CmdLine.h"
#include "mapstitch.h"

using namespace TCLAP;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  bool verbose = false;
  string outfile = "";
  vector<string> infiles;
  float max_distance = 5.;

  // Wrap everything in a try block.  Do this every time, 
  // because exceptions will be thrown for problems. 
  try {
  CmdLine cmd(
      "Aligns multiple scan maps and combines into a single image. "
      "All images given on the command line will be aligned to the first "
      "supplied image and their respective rotation/translation and "
      "transformation matrix will be returned.",
      ' ', "0.1");

  ValueArg<float> maxDistanceOpt("d","maximum-distance", "maximum distance on matched points pairs for inclusion", false,
                  5., "max-distance", cmd);
  SwitchArg verboseOpt("v","verbose","verbose output", cmd, false);
  ValueArg<string> outputFOpt("o","outfile","output filename", false,
                  "", "string",cmd);
  UnlabeledMultiArg<string> multi("fileName", "input file names (first one is pivot element)", false, "file1 and file2", cmd);

  cmd.parse( argc, argv );

  // Get the value parsed by each arg.
  verbose = verboseOpt.getValue();
  infiles = multi.getValue();
  outfile = outputFOpt.getValue();
  max_distance = maxDistanceOpt.getValue();

  } catch (ArgException &e)  // catch any exceptions
  { cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
    exit(-1);
  }

  // now let's do the real work
  if (infiles.size() != 2) {
    cerr << "error: need exactly two input files" << endl;
    exit(-1);
  }

  // load the images
  vector<Mat> images;
  for (size_t i=0; i<infiles.size(); i++) {
    Mat image = imread(infiles[i].c_str(), 0); // 0=grayscale
    if (!image.data) {
      cerr << "error: image " << infiles[i] << " not loadable." << endl;
      exit(-1);
    }
    images.push_back(image);
  }

  // create the stitched map
  StitchedMap map(images[0],images[1], max_distance);

  // write to outfile if applicable
  if (outfile.size() != 0) {
    imwrite(outfile, map.get_stitch());
  }

  if (outfile.size() == 0 || verbose) { // generate some output
    cout << "rotation: "          << map.rotation << endl
         << "translation (x,y): " << map.transx << ", " << map.transy << endl
         << "matrix: "            << map.H << endl;
  }

  if (verbose) {
    namedWindow("wrap"); imshow("wrap", map.get_stitch());
    namedWindow("debug"); imshow("debug", map.get_debug());

    while ((waitKey(0)&255) != 10) // RETURN
      ;
  }

  return 0;
}
