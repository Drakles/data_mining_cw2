from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from skimage.transform import probabilistic_hough_line


# Create the images after applying operations defined in question 1
def task1():
    # load the image
    avengers = io.imread('data/image_data/avengers_imdb.jpg')
    print('the size of the image is: ' + str(avengers.shape))

    # convert the image to gray scale
    grayscale_avengers = rgb2gray(avengers)
    # save the image
    io.imsave('outputs/avengers_grayscale.jpg', grayscale_avengers)

    # calculate treshold
    threshold = threshold_mean(grayscale_avengers)
    # convert the image to black and white based on treshold
    black_white_avengers = grayscale_avengers > threshold
    # save the image
    io.imsave('outputs/black_white.jpg', black_white_avengers)


# Create the images after applying operations defined in question 2
def task2():
    # load the image
    bush_house = io.imread('data/image_data/bush_house_wikipedia.jpg')

    # apply gaussian random noise
    bush_house_gausian_random_noise = random_noise(bush_house, var=0.1)
    # save the image
    io.imsave('outputs/bh_gaussian_random_noise.jpg',
              bush_house_gausian_random_noise)

    # apply gaussian mask
    bush_house_filtered = gaussian(bush_house_gausian_random_noise,
                                   sigma=1, multichannel=True)
    # save the image
    io.imsave('outputs/bh_gaussian_mask_filtered.jpg', bush_house_filtered)

    # apply uniform smoothing mask
    bush_house_smoothed = ndimage.\
        uniform_filter(bush_house_gausian_random_noise, size=(9, 9, 1))
    # save the image
    io.imsave('outputs/bh_gaussian_mask_filtered_smoothed.jpg',
              bush_house_smoothed)


# Create the images after applying operations defined in question 3
def task3():
    # load the image
    original_img = io.imread('data/image_data/forestry_commission_gov_uk.jpg')

    # adjust the shape
    img = original_img.reshape((-1, 3))

    # convert to np.float32
    img = np.float32(img)

    # create KMeans
    kmeans_cluster = cluster.KMeans(n_clusters=5)
    # train KMeans
    kmeans_cluster.fit(img)

    # get cluster centers and convert it to uint8
    cluster_centers = np.uint8(kmeans_cluster.cluster_centers_)

    # get labels
    cluster_labels = kmeans_cluster.labels_
    # recreate image
    img_clustered = cluster_centers[cluster_labels.flatten()]

    # save the image
    plt.imsave('outputs/forestry_kmeans.jpg',
               img_clustered.reshape(original_img.shape))


# Create the images after applying operations defined in question 4
def task4():
    # load the image
    rolland_garros = io.imread('data/image_data/rolland_garros_tv5monde.jpg')

    # apply canny edge detection
    edges = feature.canny(rgb2gray(rolland_garros))

    # apply Hough line transformation
    lines = probabilistic_hough_line(edges)

    # plot extracted lines into image
    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.xlim((0, rolland_garros.shape[1]))
    plt.ylim((rolland_garros.shape[0], 0))

    # save the image
    plt.savefig('outputs/rolland_garros.jpg')


if __name__ == '__main__':
    # task 1
    task1()

    # # task 2
    task2()

    # task 3
    task3()

    # task 4
    task4()
