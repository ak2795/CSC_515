# CSC_515
Research Project for Graduate Computer Architecture. The purpose of this project is to experimentally use GPUs to improve computer vision classification algorithm performance.

## Compiling

We developed this for mac and linux systems only. Sorry Windows !!! :)

### Mac OSX
1. Install homebrew
  https://docs.brew.sh/Installation.html
2. Install CMake or upgrade CMake
  * ```brew install cmake```
  * ```brew upgrade cmake```
3. Download or clone this repo
4. ```cd CSC_515/src```
5. ```cmake .```
6. ```make```

### Linux
1. Install CMake or upgrade CMake
  * ```apt-get install cmake```
  * ```apt-get upgrade cmake```
2. Download or clone this repo
3. ```cd CSC_515/src```
4. ```cmake .```
5. ```make```


## Usage

* ```./main IMG_FILE FUNCTION ARGS....```

### Image Functions
The first argument after the executable is the image file you would like to
perform the computer vision algorithm on.


If you would like the image to show after the algorithm has run, the last
argument should be ```show```. By default, the images do not show upon
completion. Example: ```./main img.jpg cpuBinaryThreshold show```


1. ```gpuBinaryThreshold min max numThreads```
    * min - The minimum grayscale value that the pixels must have to turn the
    pixel white.
    * max - The maximum grayscale value that the pixels must have to turn the
    pixel white
    * numThreads - The number of threads you would like to be spawned on the GPUs


2. ```cpuBinaryThreshold min max```
    * min - The minimum grayscale value that the pixels must have to turn the
    pixel white.
    * max - The maximum grayscale value that the pixels must have to turn the
    pixel white


3. ```gpuRGBThreshold rMin rMax gMin gMax bMin bMax numThreads```
    * rMin - The minimum red component that the pixels must have to turn the
    pixel white.
    * rMax - The maximum red component that the pixels must have to turn the
    pixel white
    * gMin - The minimum green component that the pixels must have to turn the
    pixel white.
    * gMax - The maximum green component that the pixels must have to turn the
    pixel white
    * bMin - The minimum blue component that the pixels must have to turn the
    pixel white.
    * bMax - The maximum blue component that the pixels must have to turn the
    pixel white
    * numThreads - The number of threads you would like to be spawned on the GPUs


4. ```cpuRGBThreshold rMin rMax gMin gMax bMin bMax```
    * rMin - The minimum red component that the pixels must have to turn the
      pixel white.
    * rMax - The maximum red component that the pixels must have to turn the
      pixel white
    * gMin - The minimum green component that the pixels must have to turn the
      pixel white.
    * gMax - The maximum green component that the pixels must have to turn the
      pixel white
    * bMin - The minimum blue component that the pixels must have to turn the
      pixel white.
    * bMax - The maximum blue component that the pixels must have to turn the
      pixel white


5. ```gpuKnearest1 k numthreads```


    This function is much faster than the gpuKNearest2. We included gpuKNearest2
    for comparison purposes.
    * k - The number of nearest neighbors you would like to compare the pixel to
    * numThreads - The number of threads you would like to be spawned on the GPUs

5. ```gpuKnearest2 k numthreads```
    * k - The number of nearest neighbors you would like to compare the pixel to
    * numThreads - The number of threads you would like to be spawned on the GPUs

5. ```cpuKnearest k```
    * k - The number of nearest neighbors you would like to compare the pixel to


## Authors

* **Joey Angeja** - *Initial work* - [jangeja94](https://github.com/jangeja94)
* **Andrew Kim** - *Initial work* - [ak2795](https://github.com/ak2795)

See also the list of [contributors](https://github.com/jangeja94/SqliteCaching/contributors) who participated in this project.

## Acknowledgments

* Professor Seng
* Cal Poly
* Friends and Family
