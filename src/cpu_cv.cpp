#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;
using namespace cv;

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
   
    // Testing for correct image reads
    /*
    imshow("Mask", mask);
    imshow("Image", image);
    imshow("Result", result);
    waitKey(0);
    */

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
    
    // Testing for correct image reads
    
    imshow("Mask", invMask);
    imshow("Image", image);
    imshow("Result", result);
    waitKey(0);
    
    
    return result;
}

int main(int argc, char** argv) {
    Mat test;

    test = maskImg("PartA/s1.jpg", "PartA_Masks/s1.jpg");
    test = maskImgInverted("PartA/s1.jpg", "PartA_Masks/s1.jpg");

    return 0; 
}
