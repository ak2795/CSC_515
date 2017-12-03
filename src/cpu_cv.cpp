#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>

using namespace std;
using namespace cv;

// Structure for a Pixel
struct Pixel {
    int cluster_id;
    unsigned char r, g, b;
    unsigned int distance;
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
vector<Pixel> createTrainingSet() {
    vector<Pixel> trainingSet;
    Mat strawberry, nonStrawberry;
    String imFile, mskFile;

    // For every image in the training folder mask images for strawberry and non-strawberry pixels
    // and add them to the training set vector
    for (int i = 0; i < 1; i++) {
        cout << "Training image " + to_string(i + 1) << endl;
        imFile = "PartA/s" + to_string(i + 10) + ".jpg";
        mskFile = "PartA_Masks/s" + to_string(i + 10) + ".jpg";
        strawberry = maskImg(imFile, mskFile);
        nonStrawberry = maskImgInverted(imFile, mskFile);
        

        // ********** STRAWBERRY PIXELS ***********
        // split the r, g, and b channels and create new pixel object
        int sRows = strawberry.rows;
        int sCols = strawberry.cols;

        unsigned int r = 0, g = 0, b = 0; // to keep track of the rgb values

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
    cout << "Training Set Size: " + to_string(trainingSet.size()) + " pixels"  << endl;
    
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
                            unsigned int r, unsigned int g, unsigned int b) {
    
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


int main(int argc, char** argv) {
    Mat input;
    vector<Pixel> trainingSet;
    int sRows, sCols;
    // TIMER VARIABLES
    double progStart, progEnd;
    double trainingStart, trainingEnd;
    double classStart, classEnd;
    double pixelStart, pixelEnd;
    double pixelSum = 0, pixelCount = 0;

    progStart = clock();
    // Read in input image to be classified
    // Check if there is an input image from the user
    if (argc < 2) {
        cout << "Must input image file" << endl;
        return -1;
    }

    input = imread(argv[1]);
    // Check if the input has valid image data
    if (!input.data) {
        cout << "Input does not hold valid data" << endl;
        return -1;
    }
    
    sRows = input.rows;
    sCols = input.cols;
    cout << "Dimensions: " + to_string(sCols) + "x" + to_string(sRows) << endl;

    // Create the training set
    cout << "Creating training set" << endl;
    trainingStart = clock();
    trainingSet = createTrainingSet();
    trainingEnd = clock();
    // Print out the training set creation duration
    cout << "Training Set Creation: " + to_string(getDuration(trainingStart, trainingEnd)) + "s" << endl;

    // Create the result matrix with the same dimensions as the input image
    Mat result(cv::Size(sCols, sRows), CV_8UC3);
    result = 0; // Initialize the matrix to all 0

    unsigned int r = 0, g = 0, b = 0; // to keep track of the rgb values
    // Classify each pixel in the input image
    classStart = clock();
    for (int j = 0; j < sRows; j++) {
        for (int k = 0; k < sCols; k++) {
            // Classify each pixel one at a time
            pixelStart = clock();
            // Get the current pixel values of the input image
            Vec3b color = input.at<Vec3b>(Point(k, j));
            b = color[0];
            g = color[1];
            r = color[2];
            
            // Classify the pixel after reading the r, g, and b values
            // K = 3
            int classification = classifyPixel(trainingSet, 3, r, g, b);
            if (classification) {
                // Set the pixel on the result matrix
                result.at<Vec3b>(Point(k, j)) = color;
                // Print out the position of the strawberry classified pixels
                // String position = "(" + to_string(k) + "," + to_string(j) + ")";
                // cout << "Location: " + position << endl;
            }
            // Timing calculation
            pixelEnd = clock();
            pixelSum += getDuration(pixelStart, pixelEnd);
            pixelCount++;
        }
    }

    cout << "Average Single Pixel Classification: " + to_string(pixelSum/pixelCount) + "s" << endl;
    classEnd = clock();
    cout << "Total Pixel Classification: " + to_string(getDuration(classStart, classEnd)) + "s"  << endl;

    // Write image to results
    // CHANGE THE FILENAME PER TEST
    imwrite("Results/bts_c1k3.jpg", result);
    //imshow("Result", result);
    //waitKey(0);
    progEnd = clock();
    cout << "Program Runtime: " + to_string(getDuration(progStart, progEnd)) + "s" << endl;
    return 0; 
}
