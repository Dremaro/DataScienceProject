"""
In order to eliminate the edges of the image and areas that are too noisy, segmentation is
necessary. It is based on the calculation of the variance of gray levels. For this purpose, the image
is divided into sub-blocks of (W × W) size’s and for each block the variance.
Then, the root of the variance of each block is compared with a threshold T, if the value obtained
is lower than the threshold, then the corresponding block is considered as the background of the
image and will be excluded by the subsequent processing.

The selected threshold value is T = 0.2 and the selected block size is W = 16

This step makes it possible to reduce the size of the useful part of the image and subsequently to
optimize the extraction phase of the biometric data.
"""
import numpy as np
import cv2 as cv


def normalise(img):
    return (img - np.mean(img))/(np.std(img))


def create_segmented_and_variance_images(im, w, threshold=.2):
    """
    Returns mask identifying the ROI. Calculates the standard deviation in each image block and threshold the ROI
    It also normalises the intesity values of
    the image so that the ridge regions have zero mean, unit standard
    deviation.
    :param im: Image
    :param w: size of the block
    :param threshold: std threshold
    :return: segmented_image
    """
    (y, x) = im.shape
    threshold = np.std(im)*threshold

    # variables initialization
    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold, gives a binary image
    mask[image_variance < threshold] = 0
    # In the end, if the image is very sharp mask is 0, otherwise it is 1 I think but not sure

    # smooth mask with a open/close morphological filter
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(w*2, w*2)) # the kernel is a circle
    # This line applies an opening operation to the mask using the kernel.
    # Opening is a morphological operation that consists of an erosion followed
    # by a dilation. It is used to remove noise, separate objects,
    # and smooth boundaries.
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # This line applies a closing operation to the mask using the kernel.
    # Closing is a morphological operation that consists of a dilation followed
    # by an erosion. It is used to close small holes, connect objects,
    # and smooth boundaries.
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # All in all, these last tow lines close holes in fingerprints lines
    # and smooth boundaries to get better delimitation.


    # normalize segmented image
    segmented_image *= mask # set to 0 the background
    im = normalise(im)
    mean_val = np.mean(im[mask==0])
    # uses the boolean mask to select only the elements from the im array that correspond to
    # True values in the mask. This creates a new array containing only the elements of im
    # where the corresponding mask value is 0.
    # Finally, np.mean(im[mask==0]) calculates the mean value of the selected elements in the im array.
    std_val = np.std(im[mask==0])
    norm_img = (im - mean_val)/(std_val)

    return segmented_image, norm_img, mask