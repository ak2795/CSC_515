

typedef struct Pixel {
    int cluster_id;
    unsigned char r, g, b;
    unsigned int distance;
} Pixel;

typedef struct RGBPixel {
  unsigned char b;
  unsigned char g;
  unsigned char r;
} RGBPixel;

void kernel simple_add(global const int* A, global const int* B,
  global int* C, const int numThreads, const int numPixels
  ) {

  C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];
}

void kernel rgbThreshold(global unsigned char* image,
  const unsigned char rMin, const unsigned char rMax, const unsigned char gMin,
  const unsigned char gMax, const unsigned char bMin, const unsigned char bMax,
  const unsigned int numThreads, const unsigned int numPixels
  ){

  unsigned int offset = get_global_id(0) * 3;
  unsigned char b, g, r;
  while (offset < numPixels * 3) {
    b = image[offset];
    g = image[offset + 1];
    r = image[offset + 2];

    if (r >= rMin && r <= rMax && g >= gMin && g <= gMax && b >= bMin && b <= bMax) {
      image[offset] = 255; image[offset + 1] = 255; image[offset + 2] = 255;
    }
    else {
      image[offset] = 0; image[offset + 1] = 0; image[offset + 2] = 0;
    }
    offset += (numThreads * 3);
  }

}

void kernel binaryThreshold(global unsigned char* image,
  const unsigned char lower, const unsigned char upper, const unsigned int numThreads,
  const unsigned int numPixels
  ) {

  unsigned int offset = get_global_id(0);
  while (offset < numPixels) {
    if (image[offset] >= lower && image[offset] <= upper) {
      image[offset] = 255;
    }
    else {
      image[offset] = 0;
    }
    offset += numThreads;
  }
}

void kernel kNearest1(global RGBPixel *image, global Pixel *trainingSet,
  const unsigned int trainingSetLength, const unsigned int numK,
  const unsigned int numThreads, const unsigned int numPixels) {

  unsigned int offset = get_global_id(0);
  float maxDistance = 0, distance;
  RGBPixel rgbPixel;
  unsigned int i, j, freq0, freq1;
  unsigned int maxLocation = 0;
  Pixel kNearestPixels[10];
  float distances[10];
  Pixel trainingPixel;
  while (offset < numPixels) {
    maxDistance = 0, distance = 0, i = 0, j = 0, freq0 = 0, freq1 = 0, maxLocation = 0;
    rgbPixel = image[offset];

    for (i = 0; i < numK; i++) {
      trainingPixel = trainingSet[i];
      distances[i] = sqrt((float)(rgbPixel.r - trainingPixel.r) * (rgbPixel.r - trainingPixel.r) +
                    (float)(rgbPixel.g - trainingPixel.g) * (rgbPixel.g - trainingPixel.g) +
                    (float)(rgbPixel.b - trainingPixel.b) * (rgbPixel.b - trainingPixel.b));
      kNearestPixels[i] = trainingSet[i];
      if (distances[i] > maxDistance) {
        maxDistance = distances[i];
        maxLocation = i;
      }
    }

    for (i = numK; i < trainingSetLength; i++) {
      trainingPixel = trainingSet[i];
      distance = sqrt((float)(rgbPixel.r - trainingPixel.r) * (rgbPixel.r - trainingPixel.r) +
                    (float)(rgbPixel.g - trainingPixel.g) * (rgbPixel.g - trainingPixel.g) +
                    (float)(rgbPixel.b - trainingPixel.b) * (rgbPixel.b - trainingPixel.b));
      if (distance < maxDistance) {
        kNearestPixels[maxLocation] = trainingSet[i];
        maxDistance = distance;
        for (j = 0; j < numK; j++) {
          if (distances[j] > maxDistance) {
            maxDistance = distances[j];
            maxLocation = j;
          }
        }
      }

    }
    freq0 = 0;
    freq1 = 0;

    for (i = 0; i < numK; i++) {
      if (kNearestPixels[i].cluster_id == 0)
        freq0++;
      else if (kNearestPixels[i].cluster_id == 1)
        freq1++;
    }
    if (freq1 > freq0) {
      image[offset].r = 255;
      image[offset].g = 255;
      image[offset].b = 255;
    }
    else {
      //image[offset].r = 0;
      //image[offset].g = 0;
      //image[offset].b = 0;
    }
    offset += numThreads;
  }
}



void kernel kNearest2(const RGBPixel pixel, global Pixel *trainingSet,
  const unsigned int trainingSetLength, const unsigned int numThreads) {

  unsigned int offset = get_global_id(0);
  float maxDistance = 0, distance;
  unsigned int i, j;
  Pixel trainingPixel;
  while (offset < trainingSetLength) {
    trainingPixel = trainingSet[offset];
    trainingSet[offset].distance = sqrt((float)(pixel.r - trainingPixel.r) * (pixel.r - trainingPixel.r) +
                  (float)(pixel.g - trainingPixel.g) * (pixel.g - trainingPixel.g) +
                  (float)(pixel.b - trainingPixel.b) * (pixel.b - trainingPixel.b));
    offset += numThreads;
  }
}
