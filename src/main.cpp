
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/imgproc.hpp>
#include <sys/time.h>
#include <sys/resource.h>
#include "cl.hpp"
#include <vector>
#include <initializer_list>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cl;

#include <fstream>
#include <streambuf>
#define MAXROWS 5000
#define MAXCOLS 5000
#define MAX_TRAININGSETS 1000

/*
 * Apply the mask to the image and output the result
 * */

 // Structure for a Pixel
 struct Pixel {
     int cluster_id;
     unsigned char r, g, b;
     unsigned int distance;
 };
 typedef struct RGBPixel {
   unsigned char b;
   unsigned char g;
   unsigned char r;
 } RGBPixel;

 struct GPUData {
   Context context;
   Program::Sources sources;
   Device device;
   Program program;
   Buffer bufferImage;
   Buffer bufferTrainingSet;
   Buffer bufferNeighbors;
   Buffer bufferBigTrainingSet;
   Buffer bufferBigNeighbors;
   vector<Pixel> trainingSet;
   Pixel *bigTrainingSet;
   CommandQueue queue;
 };

 // Get the time difference given the start and the end
 double getDuration(double t1, double t2) {
     double duration = t2 - t1;
     return duration / CLOCKS_PER_SEC;
 }

 /*
  * Apply the mask to the image and output the result
  * */
 Mat maskImg(String imPath, String mskPath) {
     Mat mask, image, result;

     // Read in the mask and the image given the paths
     mask = imread(mskPath);
     image = imread(imPath);
     // Grayscale the mask
     cvtColor(mask, mask, COLOR_RGB2GRAY);

     // Bitwise AND the mask and image
     bitwise_and(image, image, result, mask=mask);

     return result;
 }


 /*
  * Apply the inverted version of the mask to the image and output the result
  * */
 Mat maskImgInverted(String imPath, String mskPath) {
     Mat mask, invMask, image, result;

     // Read in the mask and the image given the paths
     mask = imread(mskPath);
     image = imread(imPath);
     // Grayscale the mask
     cvtColor(mask, mask, COLOR_RGB2GRAY);
     // Invert the mask
     threshold(mask, invMask, 0, 255, 1);

     // Bitwise AND the mask and image
     bitwise_and(image, image, result, mask=invMask);

     return result;
 }

 /*
  * Create the training set using the strawberry and non-strawberry pixels
  * */
 vector<Pixel> createTrainingSet(GPUData gpuData) {
     vector<Pixel> trainingSet;
     Mat strawberry, nonStrawberry;
     String imFile, mskFile;

     // For every image in the training folder mask images for strawberry and non-strawberry pixels
     // and add them to the training set vector
     for (int i = 0; i < 1; i++) {
         // cout << "Training image " + to_string(i + 1) << endl;
         imFile = "PartA/s" + to_string(i + 10) + ".jpg";
         mskFile = "PartA_Masks/s" + to_string(i + 10) + ".jpg";
         strawberry = maskImg(imFile, mskFile);
         nonStrawberry = maskImgInverted(imFile, mskFile);


         // ********** STRAWBERRY PIXELS ***********
         // split the r, g, and b channels and create new pixel object
         int sRows = strawberry.rows;
         int sCols = strawberry.cols;

         unsigned char r = 0, g = 0, b = 0; // to keep track of the rgb values

         for (int j = 0; j < sRows; j++) {
             for (int k = 0; k < sCols; k++) {
                 Vec3b color = strawberry.at<Vec3b>(Point(k, j));
                 b = color[0];
                 g = color[1];
                 r = color[2];
                 if (b || g || r) {
                     // Create a Pixel object
                     Pixel pixel = {1, r, g, b, 0};

                     // Add new pixel to the vector
                     trainingSet.push_back(pixel);
                 }
                 b = g = r = 0;
             }
         }

         // ********** NON-STRAWBERRY PIXELS ***********
         // split the r, g, and b channels and create new pixel object
         sRows = nonStrawberry.rows;
         sCols = nonStrawberry.cols;

         r = 0, g = 0, b = 0; // to keep track of the rgb values

         for (int j = 0; j < sRows; j++) {
             for (int k = 0; k < sCols; k++) {
                 Vec3b color = nonStrawberry.at<Vec3b>(Point(k, j));
                 b = color[0];
                 g = color[1];
                 r = color[2];
                 if (b || g || r) {
                     // Create a Pixel object
                     Pixel pixel = {0, r, g, b, 0};

                     // Add new pixel to the vector
                     trainingSet.push_back(pixel);
                 }
                 b = g = r = 0;
             }
         }
     }
     // cout << "Training Set Size: " + to_string(trainingSet.size()) + " pixels"  << endl;

     return trainingSet;
 }

 /*
  * Comparison between Points for sorting
  * */
 bool comparison(Pixel a, Pixel b) {
     bool result = (a.distance < b.distance);
     return result;
 }

 /*
  * Return the Euclidean distance between two sets of rgb values
  **/
 unsigned int findDistance(unsigned int r1, unsigned int g1, unsigned int b1,
                            unsigned int r2, unsigned int g2, unsigned int b2) {
     unsigned int distance;
     distance = sqrt((r2 - r1) * (r2 - r1) +
                     (g2 - g1) * (g2 - g1) +
                     (b2 - b1) * (b2 - b1));

     return distance;
 }

 /*
  * Do a comparision for the given pixel against every Pixel object in
  * */
 unsigned int classifyPixel(vector<Pixel> trainingSet, int k,
                             unsigned char r, unsigned char g, unsigned char b) {

     unsigned int n = trainingSet.size();
     // Go through the training set and find the distance between the input r, g, b
     for (int i = 0; i < n; i++) {
         trainingSet[i].distance = findDistance(trainingSet[i].r, trainingSet[i].g, trainingSet[i].b, r, g, b);
     }

     // Sort the pixels by distance (shortest first)
     sort(trainingSet.begin(), trainingSet.end(), comparison);

     // Keep track of the frequencies for each classification
     int freq0 = 0;
     int freq1 = 0;

     for (int i = 0; i < k; i++) {

         if (trainingSet[i].cluster_id == 0)
             freq0++;
         else if (trainingSet[i].cluster_id == 1)
             freq1++;
     }
     return (freq0 > freq1 ? 0 : 1);
 }

GPUData initGPU(uint numThreads) {
  GPUData gpuData;

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.size()==0) {
      std::cout<<" No platforms found. Check OpenCL installation!\n";
      exit(1);
  }
  cl::Platform default_platform=all_platforms[0];
  std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size()==0){
      std::cout<<" No devices found. Check OpenCL installation!\n";
      exit(1);
  }

  // use device[1] because that's a GPU; device[0] is the CPU
  gpuData.device = all_devices[2];
  std::cout<< "Using device: "<<gpuData.device.getInfo<CL_DEVICE_NAME>()<<"\n";

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  gpuData.context = Context({gpuData.device});

  // create the program that we want to execute on the device

  std::ifstream t("kernelCode.cl");
  std::string kernel_code((std::istreambuf_iterator<char>(t)),
                   std::istreambuf_iterator<char>());

  gpuData.sources.push_back({kernel_code.c_str(), kernel_code.length()});

  gpuData.program = Program(gpuData.context, gpuData.sources);
  if (gpuData.program.build({gpuData.device}) != CL_SUCCESS) {
      std::cout << "Error building: " << gpuData.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpuData.device) << std::endl;
      exit(1);
  }
  gpuData.bufferImage = Buffer(gpuData.context, CL_MEM_READ_WRITE, sizeof(uchar) * MAXROWS * MAXCOLS * 3);
  gpuData.queue = CommandQueue(gpuData.context, gpuData.device);

  gpuData.trainingSet = createTrainingSet(gpuData);
  gpuData.bufferTrainingSet = Buffer(gpuData.context, CL_MEM_READ_WRITE, sizeof(Pixel) * gpuData.trainingSet.size());
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferTrainingSet, CL_TRUE, 0, sizeof(Pixel) * gpuData.trainingSet.size(), gpuData.trainingSet.data());
  gpuData.bufferBigTrainingSet = Buffer(gpuData.context, CL_MEM_READ_WRITE, sizeof(Pixel) * gpuData.trainingSet.size() * MAX_TRAININGSETS);
  gpuData.bigTrainingSet = new Pixel[MAX_TRAININGSETS * gpuData.trainingSet.size()];
  for (int i = 0; i < MAX_TRAININGSETS ; i++) {
    gpuData.queue.enqueueWriteBuffer(gpuData.bufferBigTrainingSet, CL_TRUE, i * gpuData.trainingSet.size() * sizeof(Pixel), sizeof(Pixel) * gpuData.trainingSet.size(), gpuData.trainingSet.data());
  }

  return gpuData;
}




void gpu() {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.size()==0) {
      std::cout<<" No platforms found. Check OpenCL installation!\n";
      exit(1);
  }
  cl::Platform default_platform=all_platforms[0];
  std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size()==0){
      std::cout<<" No devices found. Check OpenCL installation!\n";
      exit(1);
  }

  // use device[1] because that's a GPU; device[0] is the CPU
  cl::Device default_device=all_devices[2];
  std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  cl::Context context({default_device});

  // create the program that we want to execute on the device
  cl::Program::Sources sources;

  // calculates for each element; C = A + B
  std::string kernel_code=
  "   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
   "       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
   "   }                                                                               ";
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if (program.build({default_device}) != CL_SUCCESS) {
      std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
      exit(1);
  }

  // apparently OpenCL only likes arrays ...
  // N holds the number of elements in the vectors we want to add
  int N[1] = {1000000000};
  int n = N[0];

  // create buffers on device (allocate space on GPU)
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_N(context, CL_MEM_READ_ONLY,  sizeof(int));

  // create things on here (CPU)
  int *A = new int[n];
  int *B = new int[n];
  int *C = new int[n];

  for (int i=0; i<n; i++) {
      A[i] = i;
      B[i] = n - i - 1;
  }

  // create a queue (a queue of commands that the GPU will execute)
  cl::CommandQueue queue(context, default_device);

  // push write commands to queue
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);
  queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int),   N);

  // RUN ZE KERNEL
  cl::Kernel simple_add(program, "simple_add");
  simple_add.setArg(0, buffer_A);
  simple_add.setArg(1, buffer_B);
  simple_add.setArg(2, buffer_C);
  queue.enqueueNDRangeKernel(simple_add, cl::NullRange,cl::NDRange(n),cl::NullRange);
  queue.finish();

  // read result from GPU to here
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);
  cout << C[0]<< endl;
}

void cpu() {
  int n = 1000000000;
  int *A = new int[n];
  int *B = new int[n];
  int *C = new int[n];


  for (int i=0; i<n; i++) {
      A[i] = i;
      B[i] = n - i - 1;
  }
  for (int i=0; i<n; i++) {
      C[i] = A[i] + B[i];
  }
  cout << C[0] << endl;

}

Mat cpuRGBThreshold(Mat img, uchar rMin, uchar rMax, uchar gMin, uchar gMax, uchar bMin, uchar bMax) {
  double threshStart, threshEnd;
  int rows = img.rows, cols = img.cols;
  threshStart = clock();

  for (int i = 0; i < rows * cols; i++) {
    unsigned int offset = i * 3;
    unsigned char b = img.data[offset];
    unsigned char g = img.data[offset + 1];
    unsigned char r = img.data[offset + 2];
    if (r >= rMin && r <= rMax && g >= gMin && g <= gMax && b >= bMin && b <= bMax) {
      img.data[offset] = 255;
      img.data[offset + 1] = 255;
      img.data[offset + 2] = 255;
    }
    else {
      img.data[offset ] = 0;
      img.data[offset + 1] = 0;
      img.data[offset + 2] = 0;
    }
  }
  threshEnd = clock();

  cout << "Total Pixel Thresholding: " + to_string(getDuration(threshStart, threshEnd)) << endl;

  return img;
}

Mat gpuRGBThreshold(Mat img, uchar rMin, uchar rMax, uchar gMin, uchar gMax, uchar bMin, uchar bMax, uint numThreads, GPUData gpuData) {
  double threshStart, threshEnd;
  int rows = img.rows, cols = img.cols;
  threshStart = clock();

  Kernel rgbThreshold(gpuData.program, "rgbThreshold");
  rgbThreshold.setArg(0, gpuData.bufferImage);
  rgbThreshold.setArg(1, rMin);
  rgbThreshold.setArg(2, rMax);
  rgbThreshold.setArg(3, gMin);
  rgbThreshold.setArg(4, gMax);
  rgbThreshold.setArg(5, bMin);
  rgbThreshold.setArg(6, bMax);
  rgbThreshold.setArg(7, numThreads);
  rgbThreshold.setArg(8, rows * cols);

   // push write commands to queue
   gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);

   gpuData.queue.enqueueNDRangeKernel(rgbThreshold, NullRange, numThreads, NullRange);
   gpuData.queue.finish();

   // read result from GPU to here
   gpuData.queue.enqueueReadBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar)* rows * cols * 3, img.data);

   threshEnd = clock();
   cout << "Total Pixel Thresholding: " + to_string(getDuration(threshStart, threshEnd)) << endl;

   return img;
}

Mat cpuBinaryThreshold(Mat img, uchar lowerBound, uchar upperBound) {
  double threshStart, threshEnd;
  int rows = img.rows, cols = img.cols;
  threshStart = clock();

  for (int i = 0; i < rows * cols; i++) {
    if (img.data[i] >= lowerBound && img.data[i] <= upperBound) {
      img.data[i] = 255;
    }
    else {
      img.data[i] = 0;
    }
  }
  threshEnd = clock();
  cout << "Total Pixel Thresholding: " + to_string(getDuration(threshStart, threshEnd)) << endl;
  return img;
}

Mat gpuBinaryThreshold(Mat img, uchar lowerBound, uchar upperBound, uint numThreads, GPUData gpuData) {
  double threshStart, threshEnd;
  int rows = img.rows, cols = img.cols;
  threshStart = clock();

  Kernel binaryThreshold(gpuData.program, "binaryThreshold");
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols, img.data);

  binaryThreshold.setArg(0, gpuData.bufferImage);
  binaryThreshold.setArg(1, lowerBound);
  binaryThreshold.setArg(2, upperBound);
  binaryThreshold.setArg(3, numThreads);
  binaryThreshold.setArg(4, rows * cols);

  gpuData.queue.enqueueNDRangeKernel(binaryThreshold, NullRange, numThreads, NullRange);
  gpuData.queue.finish();

  // read result from GPU to here
  gpuData.queue.enqueueReadBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols, img.data);
  threshEnd = clock();
  cout << "Total Pixel Thresholding: " + to_string(getDuration(threshStart, threshEnd)) << endl;
  return img;
}

Mat cpuKNearest(Mat img, vector<Pixel> trainingSet, int kNeighbors) {
  int sRows, sCols;
  double classStart, classEnd;
  classStart = clock();

  sRows = img.rows;
  sCols = img.cols;

  // Create the result matrix with the same dimensions as the input image
  Mat result(cv::Size(sCols, sRows), CV_8UC3);
  result = 0; // Initialize the matrix to all 0

  unsigned int r = 0, g = 0, b = 0; // to keep track of the rgb values

  for (int j = 0; j < sRows; j++) {
      for (int k = 0; k < sCols; k++) {
          // Get the current pixel values of the input image
          Vec3b color = img.at<Vec3b>(Point(k, j));
          b = color[0];
          g = color[1];
          r = color[2];

          // Classify the pixel after reading the r, g, and b values
          // K = 5
          int classification = classifyPixel(trainingSet, kNeighbors, r, g, b);
          if (classification) {
              // Set the pixel on the result matrix
              result.at<Vec3b>(Point(k, j)) = color;
              // Print out the position of the strawberry classified pixels
              String position = "(" + to_string(k) + "," + to_string(j) + ")";
          }
      }
  }
  classEnd = clock();
  cout << "Total Pixel Classification: " + to_string(getDuration(classStart, classEnd)) << endl;
  return result;
}

Mat FailedGpuKNearest1(Mat img, uint k, uint numThreads, GPUData gpuData) {
  int rows = img.rows, cols = img.cols;
  Kernel kNearest1(gpuData.program, "FailedkNearest1");
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);

  kNearest1.setArg(0, gpuData.bufferImage);
  kNearest1.setArg(1, gpuData.bufferTrainingSet);
  kNearest1.setArg(2, gpuData.trainingSet.size());
  kNearest1.setArg(3, k);
  kNearest1.setArg(4, numThreads);
  kNearest1.setArg(5, rows * cols);
  gpuData.queue.enqueueNDRangeKernel(kNearest1, NullRange, numThreads, NullRange);
  gpuData.queue.finish();

  // read result from GPU to here
  gpuData.queue.enqueueReadBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);
  return img;
}

Mat gpuKNearest1(Mat img, uint k, uint numThreads, GPUData gpuData) {
  int rows = img.rows, cols = img.cols;
  double readStart, readEnd;
  double classStart, classEnd;

  Kernel kNearest1(gpuData.program, "kNearest1");
  kNearest1.setArg(2, gpuData.bufferBigTrainingSet);
  kNearest1.setArg(3, gpuData.trainingSet.size());
  kNearest1.setArg(4, numThreads);
  RGBPixel *pixels = (RGBPixel *)(img.data);
  Pixel *trainingSet;
  classStart = clock();
  for (int i = 0 ; i < rows * cols; i++) {
    if (i % MAX_TRAININGSETS == 0 && i != 0) {
      gpuData.queue.finish();
      gpuData.queue.enqueueReadBuffer(gpuData.bufferBigTrainingSet, CL_TRUE, 0, sizeof(Pixel) * gpuData.trainingSet.size() * MAX_TRAININGSETS, gpuData.bigTrainingSet);
      for (int j = 0; j < MAX_TRAININGSETS; j++) {
        trainingSet = gpuData.bigTrainingSet + (j * gpuData.trainingSet.size());
        sort(trainingSet, trainingSet + gpuData.trainingSet.size(), comparison);

        // Keep track of the frequencies for each classification
        int freq0 = 0;
        int freq1 = 0;

        for (int i = 0; i < k; i++) {

            if (trainingSet[i].cluster_id == 0)
                freq0++;
            else if (trainingSet[i].cluster_id == 1)
                freq1++;
        }
        if (freq0 > freq1) {
          pixels[j + i - MAX_TRAININGSETS].r = 0;
          pixels[j + i - MAX_TRAININGSETS].g = 0;
          pixels[j + i - MAX_TRAININGSETS].b = 0;
        }
      }
    }

    kNearest1.setArg(0, pixels[i]);
    kNearest1.setArg(1, i % MAX_TRAININGSETS);

    gpuData.queue.enqueueNDRangeKernel(kNearest1, NullRange, numThreads, NullRange);
  }

  classEnd = clock();
  cout << "Total Pixel Classification: " + to_string(getDuration(classStart, classEnd)) << endl;

  return img;
}


Mat FailedgpuKNearest2(Mat img, uint k, uint numThreads, GPUData gpuData) {
  int rows = img.rows, cols = img.cols;
  Kernel kNearest2(gpuData.program, "kNearest2");
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);

  kNearest2.setArg(2, gpuData.bufferBigTrainingSet);
  kNearest2.setArg(3, gpuData.trainingSet.size());
  kNearest2.setArg(4, numThreads);

  Kernel kNearest3(gpuData.program, "FailedkNearest2Phase2");

  kNearest3.setArg(0, gpuData.bufferImage);
  kNearest3.setArg(1, gpuData.bufferBigTrainingSet);
  kNearest3.setArg(2, k);
  kNearest3.setArg(3, gpuData.trainingSet.size());
  kNearest3.setArg(4, MAX_TRAININGSETS);
  kNearest3.setArg(5, rows * cols);
  RGBPixel *pixels = (RGBPixel *)(img.data);

  for (int i = 0; i < rows * cols; i++) {
    if (i % MAX_TRAININGSETS == 0 && i != 0) {
      cout << i << endl;
      gpuData.queue.finish();
      kNearest3.setArg(6, i - MAX_TRAININGSETS);
      gpuData.queue.enqueueNDRangeKernel(kNearest3, NullRange, MAX_TRAININGSETS, NullRange);
      gpuData.queue.finish();
      gpuData.queue.enqueueReadBuffer(gpuData.bufferBigTrainingSet, CL_TRUE, 0, sizeof(Pixel) * gpuData.trainingSet.size() * MAX_TRAININGSETS, gpuData.bigTrainingSet);


    }
    kNearest2.setArg(0, pixels[i]);
    kNearest2.setArg(1, i % MAX_TRAININGSETS);
    gpuData.queue.enqueueNDRangeKernel(kNearest2, NullRange, numThreads, NullRange);

  }
  gpuData.queue.enqueueReadBuffer(gpuData.bufferImage, CL_TRUE, 0, rows * cols * sizeof(RGBPixel), img.data);

  return img;
}


Mat gpuKNearest2(Mat img, uint k, uint numThreads, GPUData gpuData) {
  int rows = img.rows, cols = img.cols;
  double readStart, readEnd;

  gpuData.bufferNeighbors = Buffer(gpuData.context, CL_MEM_READ_WRITE, sizeof(Pixel) * k);
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);

  Kernel kNearest2(gpuData.program, "kNearest1");
  kNearest2.setArg(1, gpuData.bufferTrainingSet);
  kNearest2.setArg(2, gpuData.trainingSet.size());
  kNearest2.setArg(3, numThreads);

  Kernel kNearest2Phase2(gpuData.program, "kNearest2Phase2");

  kNearest2Phase2.setArg(0, gpuData.bufferTrainingSet);
  kNearest2Phase2.setArg(1, gpuData.bufferNeighbors);
  kNearest2Phase2.setArg(2, gpuData.trainingSet.size());
  kNearest2Phase2.setArg(3, k);
  kNearest2Phase2.setArg(4, numThreads);

  RGBPixel *pixels = (RGBPixel *)(img.data);
  Pixel neighbors[k * MAX_TRAININGSETS];


  for (int i = 0; i < rows * cols; i++) {
    if (i % 1000 == 0) {
      cout << i << endl;

    }

    kNearest2.setArg(0, pixels[i]);
    gpuData.queue.enqueueNDRangeKernel(kNearest2, NullRange, numThreads, NullRange);
    gpuData.queue.finish();
    gpuData.queue.enqueueNDRangeKernel(kNearest2Phase2, NullRange, numThreads, NullRange);
    gpuData.queue.finish();
    gpuData.queue.enqueueReadBuffer(gpuData.bufferNeighbors, CL_TRUE, 0, sizeof(Pixel) * k, neighbors);
    int freq0 = 0;
    int freq1 = 0;

    for (int j = 0; j < k; j++) {
        if (neighbors[j].cluster_id == 0)
            freq0++;
        else if (neighbors[j].cluster_id == 1)
            freq1++;
    }
    if (freq0 > freq1) {
      pixels[i].r = 0;
      pixels[i].g = 0;
      pixels[i].b = 0;
    }
  }
  return img;
}


Mat FailedgpuKNearest3(Mat img, uint k, uint numThreads, GPUData gpuData) {
  int rows = img.rows, cols = img.cols;
  double readStart, readEnd;

  gpuData.bufferBigNeighbors = Buffer(gpuData.context, CL_MEM_READ_WRITE, sizeof(Pixel) * k * MAX_TRAININGSETS);

  Kernel kNearest2(gpuData.program, "kNearest2");
  gpuData.queue.enqueueWriteBuffer(gpuData.bufferImage, CL_TRUE, 0, sizeof(uchar) * rows * cols * 3, img.data);

  kNearest2.setArg(2, gpuData.bufferBigTrainingSet);
  kNearest2.setArg(3, gpuData.trainingSet.size());
  kNearest2.setArg(4, numThreads);

  Kernel kNearest5Phase2(gpuData.program, "kNearest5Phase2");

  kNearest5Phase2.setArg(0, gpuData.bufferBigTrainingSet);
  kNearest5Phase2.setArg(1, gpuData.bufferBigNeighbors);
  kNearest5Phase2.setArg(2, gpuData.trainingSet.size());
  kNearest5Phase2.setArg(3, k);
  kNearest5Phase2.setArg(4, numThreads);

  RGBPixel *pixels = (RGBPixel *)(img.data);
  Pixel neighbors[k * MAX_TRAININGSETS];
  Pixel *relativeNeighbors;

  for (int i = 0 ; i < rows * cols; i++) {
    cout << i << endl;
    if (i % MAX_TRAININGSETS == 0 && i != 0) {
      gpuData.queue.finish();
      gpuData.queue.enqueueNDRangeKernel(kNearest5Phase2, NullRange, numThreads, NullRange);
      gpuData.queue.finish();
      gpuData.queue.enqueueReadBuffer(gpuData.bufferBigNeighbors, CL_TRUE, 0, sizeof(Pixel) * k, neighbors);
      for (int j = 0; j < MAX_TRAININGSETS; j++) {
        relativeNeighbors = neighbors + (j * k);
        int freq0 = 0;
        int freq1 = 0;

        for (int l = 0; l < k; l++) {
            if (relativeNeighbors[l].cluster_id == 0)
                freq0++;
            else if (relativeNeighbors[l].cluster_id == 1)
                freq1++;
        }

        if (freq0 > freq1) {
          pixels[j + i - MAX_TRAININGSETS].r = 0;
          pixels[j + i - MAX_TRAININGSETS].g = 0;
          pixels[j + i - MAX_TRAININGSETS].b = 0;
        }
      }
      cout << i << endl;
    }

    kNearest2.setArg(0, pixels[i]);
    kNearest2.setArg(1, i % MAX_TRAININGSETS);

    gpuData.queue.enqueueNDRangeKernel(kNearest2, NullRange, numThreads, NullRange);

  }

  return img;
}



int main(int argc, char** argv) {

  uint numThreads = 10000000;
  Mat rawGrayImage;
  Mat rawRGBImage;
  Mat parsedImage;


  // TIMING VARIABLES
  double progStart, progEnd;
  double trainStart, trainEnd;

  progStart = clock();

  if (argc < 2) {
   cout << "Must enter an image file" << endl;
   return -1;
  }

  rawRGBImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (!rawRGBImage.data) {
     cout << "Could not read image data" << endl;
     return -1;
  }


  cvtColor(rawRGBImage, rawGrayImage, CV_BGR2GRAY);
  if (argc < 3) {
    cout << "Must input a function.\n"\
      "\tgpuBinaryThreshold\n"\
      "\tcpuBinaryThreshold\n"\
      "\tgpuRGBThreshold\n"\
      "\tcpuRGBThreshold\n"\
      "\tgpuKNearest1\n"\
      "\tgpuKNearest2\n"\
      "\tcpuKNearest" << endl;
      return -1;
  }

  GPUData gpuData = initGPU(numThreads);

  if (!strcmp(argv[2],"gpuBinaryThreshold")) {
    if (argc < 5) {
      cout << "Additional requirements needed for gpuBinaryThreshold:\n"\
        "\tmin - Minimum grayscale value that pixels must be greater than.\n"\
        "\tmax - Maximum grayscale value that pixels must be less than.\n"\
        "\tnumThreads - Number of threads to run on the GPU.\n" << endl;
      return -1;
    }
    cout << "Running gpuBinaryThreshold" << endl;

    parsedImage = gpuBinaryThreshold(rawGrayImage, atoi(argv[3]), atoi(argv[4]),
      atoi(argv[5]), gpuData);
  }

  else if (!strcmp(argv[2],"cpuBinaryThreshold")) {
    if (argc < 4) {
      cout << "Additional requirements needed for cpuBinaryThreshold:\n"\
        "\tmin - Minimum grayscale value that pixels must be greater than.\n"\
        "\tmax - Maximum grayscale value that pixels must be less than.\n" << endl;
      return -1;
    }
    cout << "Running cpuBinaryThreshold" << endl;
    parsedImage = cpuBinaryThreshold(rawGrayImage, atoi(argv[3]), atoi(argv[4]));

  }

  else if (!strcmp(argv[2],"gpuRGBThreshold")) {
    if (argc < 9) {
      cout << "Additional requirements needed for cpuRGBThreshold:\n"\
        "\tminR - Minimum R component value that pixels must be greater than.\n"\
        "\tmaxR - Maximum R component value that pixels must be less than.\n"
        "\tminG - Minimum G component value that pixels must be greater than.\n"\
        "\tmaxG - Maximum G component value that pixels must be less than.\n"\
        "\tminB - Minimum B component value that pixels must be greater than.\n"\
        "\tmaxB - Maximum B component value that pixels must be less than.\n"\
        "\tnumThreads - Number of threads to run on the GPU.\n" << endl;
      return -1;
    }
    cout << "Running gpuRGBThreshold" << endl;
    parsedImage = gpuRGBThreshold(rawRGBImage, atoi(argv[3]), atoi(argv[4]),
      atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]),
      atoi(argv[9]), gpuData);
  }

  else if (!strcmp(argv[2],"cpuRGBThreshold")) {
    if (argc < 8) {
      cout << "Additional requirements needed for cpuRGBThreshold:\n"\
        "\tminR - Minimum R component value that pixels must be greater than.\n"\
        "\tmaxR - Maximum R component value that pixels must be less than.\n"
        "\tminG - Minimum G component value that pixels must be greater than.\n"\
        "\tmaxG - Maximum G component value that pixels must be less than.\n"\
        "\tminB - Minimum B component value that pixels must be greater than.\n"\
        "\tmaxB - Maximum B component value that pixels must be less than.\n" << endl;
      return -1;
    }
    cout << "Running cpuRGBThreshold" << endl;
    parsedImage = cpuRGBThreshold(rawRGBImage, atoi(argv[3]), atoi(argv[4]),
      atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]));
  }
  else if (!strcmp(argv[2],"gpuKNearest1")) {
    if (argc < 4) {
      cout << "Additional requirements needed for gpuKNearest1:\n"\
        "\tk - Number of k Neighbors for the classification.\n"\
        "\tnumThreads - Number of threads to run on the GPU.\n" << endl;
      return -1;
    }
    cout << "Running gpuKNearest1" << endl;
    parsedImage = gpuKNearest1(rawRGBImage, atoi(argv[3]), atoi(argv[4]), gpuData);
  }
  else if (!strcmp(argv[2],"gpuKNearest2")) {
    if (argc < 4) {
      cout << "Additional requirements needed for gpuKNearest2:\n"\
        "\tk - Number of k Neighbors for the classification.\n"\
        "\tnumThreads - Number of threads to run on the GPU.\n" << endl;
      return -1;
    }
    cout << "Running gpuKNearest2" << endl;
    parsedImage = gpuKNearest2(rawRGBImage, atoi(argv[3]), atoi(argv[4]), gpuData);
  }
  else if (!strcmp(argv[2],"cpuKNearest")) {
    if (argc < 3) {
      cout << "Additional requirements needed for cpuKNearest:\n"\
        "\tk - Number of k Neighbors for the classification.\n" << endl;
      return -1;
    }
    cout << "Running cpuKNearest" << endl;
    parsedImage = cpuKNearest(rawRGBImage, gpuData.trainingSet, atoi(argv[3]));
  }

  else {
    cout << "Invalid Selection. Valid choices are:\n"\
      "\tgpuBinaryThreshold\n"\
      "\tcpuBinaryThreshold\n"\
      "\tgpuRGBThreshold\n"\
      "\tcpuRGBThreshold\n"\
      "\tgpuKNearest1\n"\
      "\tgpuKNearest2\n"\
      "\tcpuKNearest" << endl;
    return -1;
  }

  if (!strcmp(argv[argc -1], "show")) {
    namedWindow("Threshold Test", CV_WINDOW_AUTOSIZE);
    imshow("Threshold Test", parsedImage);
    waitKey(0);
  }

  printf("done\n");
  progEnd = clock();
  cout << "Program Runtime: " + to_string(getDuration(progStart, progEnd)) << endl;
  return 0;
}
