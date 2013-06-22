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
#include <string>
#include <iostream>
#include <algorithm>
#include <opencv/highgui.h>
#include "tclap/CmdLine.h"
#include "mapstitch/mapstitch.h"

using namespace TCLAP;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
  bool verbose = false;
  string outfile = "";
  vector<string> infiles;
  float max_distance = 5.;

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
    cout << "rotation: "          << map.rot_deg << endl
         << "translation (x,y): " << map.transx << ", " << map.transy << endl
         << "matrix: "            << map.H << endl;
  }

  if (verbose) {
    /*namedWindow("wrap"); imshow("wrap", map.get_stitch());*/ imwrite("stitch.pgm", map.get_stitch());
    /*namedWindow("debug"); imshow("debug", map.get_debug());*/ imwrite("debug.pgm", map.get_debug());

  }

  return 0;
}
