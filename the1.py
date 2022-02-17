#Berk Güler 2310092
#Onur Demir 2309870

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def calculate_lookup(src_cdf, ref_cdf):
    lookup_table = np.zeros(256)
    smallest_diff_index = 0
    for src_pixel_val in range(len(src_cdf)):
        absolute_value_array = np.abs(ref_cdf-src_cdf[src_pixel_val])
        smallest_diff_index = absolute_value_array.argmin()
        lookup_table[src_pixel_val] = smallest_diff_index
    return lookup_table


def gaussianHistogram(m, s, rowSize, colSize, output_path):
    m1 = m[0]
    m2 = m[1]
    s1 = s[0]
    s2 = s[1]
    N = rowSize * colSize * 3
    data_1 = np.random.randn(N) * s1 + m1
    data_2 = np.random.randn(N) * s2 + m2
    final_data = np.concatenate((data_1, data_2))
    (n, bins, patches) = plt.hist(final_data, 256, [0, 256], color='blue')
    cdf = np.cumsum(n)
    cdf_normalized = cdf / float(cdf.max())
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(output_path + '/gaussian_histogram.png')
    plt.clf()
    return cdf_normalized


def originalHistogram(image, rowSize, colSize, output_path, primary):
    y = np.zeros(256)
    for i in range(0, rowSize):
        for j in range(0, colSize):
            y[image[i][j][0]] += 3
    x = np.arange(0, 256)
    cdf = np.cumsum(y)
    cdf_normalized = cdf / float(cdf.max())
    plt.bar(x, y, color="blue", align="center")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if primary:
        plt.savefig(output_path + '/original_histogram.png')
    else:
        plt.savefig(output_path + '/matched_image_histogram.png')
    plt.clf()
    return cdf_normalized


def part1(input_img_path, output_path, m, s):
    myImage = cv.imread(input_img_path)
    rowSize = myImage.shape[0]
    colSize = myImage.shape[1]
    new_image = np.zeros((rowSize, colSize, 3), np.uint8)
    src_cdf = originalHistogram(myImage, rowSize, colSize, output_path, 1)
    ref_cdf = gaussianHistogram(m, s, rowSize, colSize, output_path)
    look_up_table = calculate_lookup(src_cdf, ref_cdf)
    for i in range(0, rowSize):
        for j in range(0, colSize):
            new_image[i][j][0] = look_up_table[myImage[i][j][0]]
            new_image[i][j][1] = look_up_table[myImage[i][j][1]]
            new_image[i][j][2] = look_up_table[myImage[i][j][2]]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/matched_image.png', new_image)
    originalHistogram(new_image, rowSize, colSize, output_path, 0)

    return new_image


def the1_convolution(input_img_path, myFilter):
    myImage = cv.imread(input_img_path)
    rowSize = myImage.shape[0]
    colSize = myImage.shape[1]
    filterSize = len(myFilter)
    delta = filterSize // 2
    newImage = np.zeros((rowSize, colSize, 3))

    for i in range(delta, rowSize - delta):
        for j in range(delta, colSize - delta):
            pixelValR = 0
            pixelValG = 0
            pixelValB = 0
            for k in range(0, filterSize):
                for m in range(0, filterSize):
                    x = i - (delta - k)
                    y = j - (delta - m)
                    pixelValR += myImage[x][y][0] * myFilter[k][m]
                    pixelValG += myImage[x][y][1] * myFilter[k][m]
                    pixelValB += myImage[x][y][2] * myFilter[k][m]
            pixelValR = pixelValR // (filterSize * filterSize)
            pixelValG = pixelValG // (filterSize * filterSize)
            pixelValB = pixelValB // (filterSize * filterSize)
            newImage[i][j][0] = pixelValR
            newImage[i][j][1] = pixelValG
            newImage[i][j][2] = pixelValB
    return newImage


def part2(input_img_path, output_path):
    img = cv.imread(input_img_path)
    grayScaleImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussianImage = cv.GaussianBlur(grayScaleImage, (5, 5), 1.4)

    wh = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    wv = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    wld = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    wrd = np.array([[0, 1, 1], [-1, 0, -1], [-1, -1, 0]])

    pwtx = cv.filter2D(gaussianImage, -1, wh)
    pwty = cv.filter2D(gaussianImage, -1, wv)
    pwtld = cv.filter2D(gaussianImage, -1, wld)
    pwtrd = cv.filter2D(gaussianImage, -1, wrd)

    last = pwtx + pwty + pwtld + pwtrd

    last[last > 75] = 255
    last[last < 50] = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/edges.png', last)
    return last


def enhance_3(path_to_3, output_path):
    img = cv.imread(path_to_3)
    median = cv.medianBlur(img, 5)
    kernel = np.ones((5, 5), np.float32) / 25
    smoothedImage = cv.filter2D(img, -1, kernel)
    gaussianImage = cv.GaussianBlur(img, (5, 5), 1.4)
    b, g, r = cv.split(img)
    b = cv.GaussianBlur(b, (3,3), 1)
    r = cv.GaussianBlur(r, (5,5), 1.4)
    g = cv.GaussianBlur(g, (5,5), 1.4)
    last = cv.merge([b,g,r])


    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/enhanced.png', gaussianImage)


def enhance_4(path_to_4, output_path):
    img = cv.imread(path_to_4)
    median = cv.medianBlur(img, 5)
    kernel = np.ones((5, 5), np.float32) / 25
    smoothedImage = cv.filter2D(img, -1, kernel)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cv.imwrite(output_path + '/enhanced.png', median)
    return median
