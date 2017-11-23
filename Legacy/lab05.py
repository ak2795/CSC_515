import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import matplotlib
import math
import copy
import scipy
from scipy import ndimage
import matplotlib.colors as colors
from scipy import misc
import scipy.misc

# Read in an image given the filename
def read_image(filename):
    image = cv2.imread(filename)
    #convert from bgr to rgb color space
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])

    #output the rgb image
    return rgb_image

#show the image passed in
def s(img):
    plt.imshow(img, interpolation='none')

#show the grayscale of the image passed in
def sg(img):
    plt.imshow(img, interpolation='none', cmap='gray')

#apply the mask to the image and output the original the mask and the result
def mask_img(img, mask):
    msk = read_image(mask)
    msk = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)

    orig = read_image(img)
    #bitwise and the mask and the image
    res = cv2.bitwise_and(orig, orig, mask=msk)

    return orig, msk, res

# same thing as mask_img but with inverting the mask
def mask_img_inv(img, mask):
    msk = read_image(mask)
    msk = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
    #invert the colors of the mask
    ret, msk = cv2.threshold(msk, 0, 255, cv2.THRESH_BINARY_INV)

    orig = read_image(img)
    res = cv2.bitwise_and(orig, orig, mask=msk)

    return orig, msk, res

# normalize the rgb components of the image
def normalized(down):

    norm = np.zeros((len(down), len(down[0]), 3), np.float32)

    r = down[:, :, 0]
    g = down[:, :, 1]
    b = down[:, :, 2]

    s = b + g + r

    norm[:, :, 0] = (r / s)
    norm[:, :, 1] = (g / s)
    norm[:, :, 2] = (b / s)

    # set all the nans to 0
    norm[np.isnan(norm)] = 0

    return norm

#Get the normalized rgb vectors for the strawberry pixels
def GetMaskImgData (img, mask):
    orig, msk, res = mask_img(img, mask)

    color = 'r', 'g', 'b'

    #get the normalized red, blue, and green
    nr, nb, ng = cv2.split(normalized(res.astype(np.float32)))
    r = nr[np.nonzero(nr)]
    g = ng[np.nonzero(ng)]
    b = nb[np.nonzero(nb)]

    #print "Strawberry pixel mean: ", np.mean(b)
    return r, g, b

#Get the normalized rgb vectors for the non-strawberry pixels
def GetMaskImgData_Inv (img, mask):
    orig, msk, res = mask_img_inv(img, mask)

    #print the histogram
    color = 'r', 'g', 'b'

    #get the normalized red, blue, and green
    nr, ng, nb = cv2.split(normalized(res.astype(np.float32)))
    r = nr[np.nonzero(nr)]
    g = ng[np.nonzero(ng)]
    b = nb[np.nonzero(nb)]

    print "Non-strawberry pixel mean: ", np.mean(b)
    return r, g, b

#Perform nearest mean using the given parameters
def NearestMeanR(imgFile, m1, m2):
    img = read_image(imgFile)

    #get the normalized red, blue, and green for the current image
    nr, ng, nb = cv2.split(normalized(img.astype(np.float32)))

#scripts
r, g, b = GetMaskImgData("PartA/s1.JPG", "PartA_Masks/s1.JPG")
ir, ig, ib = GetMaskImgData_Inv("PartA/s1.JPG", "PartA_Masks/s1.JPG")
