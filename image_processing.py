from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_mean


def task1():
    avengers = io.imread('data/image_data/avengers_imdb.jpg')
    print('the size of the image is: ' + str(avengers.shape))
    grayscale_avengers = rgb2gray(avengers)
    io.imsave('outputs/avengers_grayscale.jpg', grayscale_avengers)
    treshold = threshold_mean(grayscale_avengers)
    black_white_avengers = grayscale_avengers > treshold
    io.imsave('outputs/black_white.jpg', black_white_avengers)


if __name__ == '__main__':
    # task1
    task1()


