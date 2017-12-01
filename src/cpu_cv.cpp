#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>

using namespace std;
using namespace cv;

// Structure for a Point
struct Point {
    int cluster_id;
    unsigned char r, g, b;
    int distance;
};

// Structure for a Pixel
struct Pixel {
    int cluster_id;
    unsigned char r, g, b;
};

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
vector<Pixel> createTrainingSet() {
    vector<Pixel> trainingSet;
    Mat strawberry, nonStrawberry;
    String imFile, mskFile;

    // For every image in the training folder mask images for strawberry and non-strawberry pixels
    // and add them to the training set vector
    for (int i = 0; i < 1; i++) {
        imFile = "PartA/s" + to_string(i + 10) + ".jpg";
        mskFile = "PartA_Masks/s" + to_string(i + 10) + ".jpg";
        strawberry = maskImg(imFile, mskFile);
        nonStrawberry = maskImgInverted(imFile, mskFile);
        

        // ********** STRAWBERRY PIXELS ***********
        // split the r, g, and b channels and create new pixel object
        int channels = strawberry.channels();
        int sRows = strawberry.rows;
        int sCols = strawberry.cols * channels;

        
        // Check if continuous
        if (strawberry.isContinuous()) {
            sCols *= sRows;
            sRows = 1;
        }
       
        unsigned char r = 0, g = 0, b = 0; // to keep track of the rgb values
        int count = 0; // keep track of columns (b = 0; g = 1; r = 2)
        unsigned char* p = strawberry.ptr<unsigned char>();

        for (int j = 0; j < sRows; j++) {
            for (int k = 0; k < sCols; k++, p++) {
                // save the values of b, g, and r regardless of value
                switch(count) {
                    case 0:
                        b = *p;
                        break;
                    case 1:
                        g = *p;
                        break;
                    case 2:
                        r = *p;
                        break;
                    default:
                        break;
                }
               
                if (count == 2) {
                    if (r || g || b) {

                        // Create a Pixel object
                        Pixel pixel = {1, r, g, b};
                        
                        // Add new pixel to the vector
                        trainingSet.push_back(pixel);
                    }
                    b = g = r = 0;
                    count = 0;
                }

                else
                    count++;
            }
        }
    }
    cout << trainingSet.size() << endl;
    return trainingSet;
}


int main(int argc, char** argv) {
    Mat strawberry, nonStrawberry;
    // Get the images for the strawberry and non-strawberry
    strawberry = maskImg("PartA/s1.jpg", "PartA_Masks/s1.jpg");
    nonStrawberry = maskImgInverted("PartA/s1.jpg", "PartA_Masks/s1.jpg");
    createTrainingSet();

    return 0; 
}
