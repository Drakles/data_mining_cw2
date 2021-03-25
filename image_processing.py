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


def task1():
    avengers = io.imread('data/image_data/avengers_imdb.jpg')
    print('the size of the image is: ' + str(avengers.shape))

    grayscale_avengers = rgb2gray(avengers)
    io.imsave('outputs/avengers_grayscale.jpg', grayscale_avengers)

    threshold = threshold_mean(grayscale_avengers)
    black_white_avengers = grayscale_avengers > threshold
    io.imsave('outputs/black_white.jpg', black_white_avengers)


def task2():
    # apply separately
    bush_house = io.imread('data/image_data/bush_house_wikipedia.jpg')

    bush_house_gausian_random_noise = random_noise(bush_house, var=0.1)
    io.imsave('outputs/bh_gaussian_random_noise.jpg', bush_house_gausian_random_noise)

    bush_house_filtered = gaussian(bush_house_gausian_random_noise, sigma=1, multichannel=True)
    io.imsave('outputs/bh_gaussian_mask_filtered.jpg', bush_house_filtered)

    # for uniform_filter you need to pass a 2D mask to get a coloured image bc otherwise it just averages all 3
    # colour channels
    bush_house_smoothed = ndimage.uniform_filter(bush_house_gausian_random_noise, size=(9, 9, 1))
    io.imsave('outputs/bh_gaussian_mask_filtered_smoothed.jpg', bush_house_smoothed)


def task3():
    original_img = io.imread('data/image_data/forestry_commission_gov_uk.jpg')
    img = original_img.reshape((-1, 3))
    # convert to np.float32
    img = np.float32(img)
    kmeans_cluster = cluster.KMeans(n_clusters=5)
    kmeans_cluster.fit(img)
    # convert to uint8
    cluster_centers = np.uint8(kmeans_cluster.cluster_centers_)
    cluster_labels = kmeans_cluster.labels_
    img_clustered = cluster_centers[cluster_labels.flatten()]
    plt.imsave('outputs/forestry_kmeans.jpg', img_clustered.reshape(original_img.shape))


def task4():
    rolland_garros = io.imread('data/image_data/rolland_garros_tv5monde.jpg')

    edges = feature.canny(rgb2gray(rolland_garros))

    lines = probabilistic_hough_line(edges)
    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.xlim((0, rolland_garros.shape[1]))
    plt.ylim((rolland_garros.shape[0], 0))

    plt.savefig('outputs/rolland_garros.jpg')


if __name__ == '__main__':
    # task1
    task1()

    # # task 2
    task2()

    # task 3
    task3()

    # task 4
    task4()

