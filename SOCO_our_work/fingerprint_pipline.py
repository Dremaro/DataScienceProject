import cv2 as cv
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from tqdm import tqdm
from utils.skeletonize import skeletonize


def open_images(directory):
    images_paths = glob(directory)
    return np.array([cv.imread(img_path,0) for img_path in images_paths])

def write_list_to_file(my_list, filename):
    with open(filename, 'w') as f:
        for item in my_list:
            # f.write("%s\n" % item)
            f.write(str(item)+'\n')

def turn_minutiae2numbers(minutiae):
    new_minutiae = []
    for point in minutiae:
        new_point=[point[0], point[1], point[2]]
        if new_point[2] == 'ending':
            new_point[2] = 1
        elif new_point[2] == 'bifurcation':
            new_point[2] = 3
        new_minutiae.append(new_point)
    return new_minutiae
        
    


def fingerprint_pipline(input_img):
    block_size = 16

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(input_img.copy(), float(100), float(100))

    # color threshold
    # threshold_img = normalized_img
    # _, threshold_im = cv.threshold(normalized_img,127,255,cv.THRESH_OTSU)
    # cv.imshow('color_threshold', normalized_img); cv.waitKeyEx()

    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)

    # orientations
    # Calculate_angles() function estimates the orientation of ridges in an image.
    # It takes an image, the width of the ridge, and a boolean indicating whether 
    # to smooth the angles as input. It uses the Sobel operator to calculate the
    # gradients of the image, then calculates the angles of the gradients in blocks 
    # of size W. If the smoth parameter is True, it applies a smoothing function to 
    # the angles to reduce noise.
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    # gives a colored images, colored depending on the orientation of the ridges
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    #print(type(angles), angles.shape, angles.dtype)


    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)

    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)

    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)

    # minutias
    (minutias, coordminutias) = calculate_minutiaes(thin_image)

    # singularities
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)



    # visualize pipeline stage by stage
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
    output_data = [coordminutias, angles, thin_image]
    
    # for i in range(len(output_imgs)):
    #     if len(output_imgs[i].shape) == 2:
    #         output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    # results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)

    # return results
    return output_imgs, output_data


datalink = {'input':0,
            'normalized':1,
            'segmented':2,
            'orientation':3,
            'gabor':4,
            'thin':5,
            'minutias':6,
            'singularities':7}
create_matching_fingerprint_dtb = True
print_output = False
normal = False


if __name__ == '__main__':
    
    if create_matching_fingerprint_dtb:
        ################ create minutiae database ################
        print('Creating minutiae database')
        # open images
        input_dtb_dir = './SOCO_our_work/sample_inputs/*'
        output_dtb_dir = './Zd_fingerprint_dtb/'

        images = open_images(input_dtb_dir)
        # image pipeline
        os.makedirs(output_dtb_dir, exist_ok=True)
        for i, img in enumerate(tqdm(images)):
            img_output_dir = os.path.join(output_dtb_dir, str(i))
            os.makedirs(img_output_dir, exist_ok=True)

            output_imgs, output_data = fingerprint_pipline(img)  # extracting treated data
            [minutiaes, angles, thin_image] = output_data        # attributing data to variables
            minutiaes = turn_minutiae2numbers(minutiaes)         # turning minutiae type to numbers (ending=1, bifurcation=3)

            #cv.imwrite(os.path.join(img_output_dir, 'image.png'), output_imgs[datalink['orientation']])
            # write_list_to_file(minutiaes, os.path.join(img_output_dir, 'minutiaes.txt'))
            np.savetxt(os.path.join(img_output_dir, '6_minutiaes.txt') , np.array(minutiaes))
            np.savetxt(os.path.join(img_output_dir, '3_angles.txt'), angles)
            np.savetxt(os.path.join(img_output_dir, '5_thin_image.txt'), thin_image)
        

        ################ create fingerprint2match ################
        print('Creating fingerprint2match')
        #open image
        input_match_dir = './SOCO_our_work/sample_2match/*'
        output_match_dir = './Z_fingerprint2match/'
        images = open_images(input_match_dir)
        # image pipeline
        os.makedirs(output_match_dir, exist_ok=True)
        for i, img in enumerate(tqdm(images)):

            output_imgs, output_data = fingerprint_pipline(img)
            [minutiaes, angles, thin_image] = output_data
            minutiaes = turn_minutiae2numbers(minutiaes)

            #cv.imwrite(output_match_dir+'2match_im.png', output_imgs[datalink['orientation']])
            # write_list_to_file(minutiaes, output_match_dir+'2match_minutiae.txt')
            np.savetxt(output_match_dir+'6_2match_minutiae.txt', np.array(minutiaes))
            np.savetxt(output_match_dir+'3_2match_angles.txt', angles)
            np.savetxt(output_match_dir+'5_2match_thin_image.txt', thin_image)



    if print_output:
        # open images
        img_dir = './SOCO_our_work/sample_inputs/*'
        output_dir = './SOCO_our_work/output/'

        images = open_images(img_dir)
        # image pipeline
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(tqdm(images)):
            output_imgs, outpu_data = fingerprint_pipline(img)
            fig, axs = plt.subplots(2, 4, figsize=(15, 15))
            for j, ax in enumerate(axs.flatten()):
                if j < len(output_imgs):
                    ax.imshow(output_imgs[j], cmap='gray')
                    ax.set_title(list(datalink.keys())[j])
                else:
                    ax.axis('off')
            plt.savefig(output_dir+str(i)+'.png')
    
    if normal:
        # open images
        img_dir = './SOCO_our_work/sample_inputs/*'
        output_dir = './SOCO_our_work/output/'

        images = open_images(img_dir)

        # image pipeline
        os.makedirs(output_dir, exist_ok=True)
        for i, img in enumerate(tqdm(images)):
            results = fingerprint_pipline(img)
            cv.imwrite(output_dir+str(i)+'.png', results)
            # cv.imshow('image pipeline', results); cv.waitKeyEx()


