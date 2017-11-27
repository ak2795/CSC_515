#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;
using namespace cv;

/*
 * This program runs kmeans clustering.
 * Generates ane image with random points, then assigns a random number of cluster centers
 * and uses kmeans to move those cluster centers to the correct location. *
 * */

int kmeans_test() {
   struct rusage usage;
   struct timeval start, end;

   getrusage(RUSAGE_SELF, &usage);
   start = usage.ru_utime;

   const int MAX_CLUSTERS = 5;
   Scalar colorTab[] = {
      Scalar (0, 0, 255),
      Scalar (0, 255, 0),
      Scalar (255, 100, 100),
      Scalar (255, 0, 255),
      Scalar (255, 255, 0)
   };

   Mat img(500, 500, CV_8UC3);
   RNG rng(12345);

   for (;;) {
      int k, clusterCount = rng.uniform(2, MAX_CLUSTERS);
      int i, sampleCount = rng.uniform(1, 1001);
      Mat points(sampleCount, 1, CV_32FC2), labels;

      clusterCount = MIN(clusterCount, sampleCount);
      Mat centers;

      /* Generate a random sample from multigaussian distribution */
      for (k = 0; k < clusterCount; k++) {
           Point center;
           center.x = rng.uniform (0, img.cols);
           center.y = rng.uniform (0, img.rows);
           Mat pointChunk = points.rowRange(k * sampleCount / clusterCount, k == clusterCount - 1 ?
                                            sampleCount : (k + 1) * sampleCount / clusterCount);
           rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.x), Scalar(img.cols * 0.05, img.rows * 0.05));
      }

      randShuffle(points, 1, &rng);
      kmeans(points, clusterCount, labels,
           TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
              3, KMEANS_PP_CENTERS, centers);
      img = Scalar::all(0);
      for( i = 0; i < sampleCount; i++ )
      {
           int clusterIdx = labels.at<int>(i);
           Point ipt = points.at<Point2f>(i);
           circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
      }
      imshow("clusters", img);
      char key = (char)waitKey();


      getrusage(RUSAGE_SELF, &usage);
      end = usage.ru_utime;

      if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
           break;
   }

   return 0;
}

int main(int argc, char** argv) {
   if (argc != 2) {
      cout << "Must enter an image file" << endl;
      return -1;
   }

   Mat src, src_gray, dst;
   src = imread(argv[1], CV_LOAD_IMAGE_COLOR);

   if (!src.data) {
      cout << "Could not read image data" << endl;
      return -1;
   }

   // change to grayscale image
   cvtColor(src, src_gray, CV_BGR2GRAY);

   // set threshold
   threshold(src_gray, dst, 100, 255, 0);

   // create window and display it
   namedWindow("Threshold Test", CV_WINDOW_AUTOSIZE);
   imshow("Threshold Test", dst);
   waitKey(0);

   return 0;
}
