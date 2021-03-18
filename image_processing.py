from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.util import random_noise
from skimage.filters import gaussian


def task1():
    avengers = io.imread('data/image_data/avengers_imdb.jpg')
    print('the size of the image is: ' + str(avengers.shape))

    grayscale_avengers = rgb2gray(avengers)
    io.imsave('outputs/avengers_grayscale.jpg', grayscale_avengers)

    threshold = threshold_mean(grayscale_avengers)
    black_white_avengers = grayscale_avengers > threshold
    io.imsave('outputs/black_white.jpg', black_white_avengers)


if __name__ == '__main__':
    # task1
    task1()

    # task 2
    bush_house = io.imread('data/image_data/bush_house_wikipedia.jpg')

    bush_house_gausian_random_noise = random_noise(bush_house, var=0.1)
    io.imsave('outputs/bh_gaussian_random_noise.jpg', bush_house_gausian_random_noise)

    bush_house_filtered = gaussian(bush_house_gausian_random_noise,sigma=1)
    io.imsave('outputs/bh_gaussian_mask_filtered.jpg', bush_house_filtered)

