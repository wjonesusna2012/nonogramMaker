import numpy as np
import cv2 as cv
import random
from functools import reduce

def is_grayscale(my_image):
    return len(my_image.shape) < 3

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value

def calculatePixelValue(sourceMatrix, transformationMatrix, sourceX, sourceY, scalingFactor):
    sum = 0
    for i in range((scalingFactor // 2) * -1, scalingFactor // 2):        
        for j in range((scalingFactor // 2) * -1, scalingFactor // 2):
            sum += transformationMatrix[i + scalingFactor // 2][j + scalingFactor // 2] * sourceMatrix[i + sourceX][j + sourceY]
    return sum

def transformOnIntensity(my_image): 
    height, width, n = my_image.shape
    transformedImage = [[0 for i in range(width)] for j in range(height)]

    for i in range (0, width): 
        for j in range(0, height):
            sum = 0
            for k in range(0, n):
                sum += my_image[i, j, k]
            transformedImage[i][j] = saturated(sum // n)
    return transformedImage

def generateLinearTransformationMatrix(scalingFactor):
    return [[1/(scalingFactor ** 2)] * scalingFactor for x in range (scalingFactor)]

def blockify(my_image, n):
    if is_grayscale(my_image):
        height, width = my_image.shape
        imageToTransform = my_image
    else:
        my_image = cv.cvtColor(my_image, cv.CV_8U)
        height, width, n_channels = my_image.shape
        imageToTransform = transformOnIntensity(my_image)
    scalingFactor = min(height, width) // n
    scalingFactor = 100 // 20
    result = [[0] * n for x in range(n)]
    tMatrix = generateLinearTransformationMatrix(scalingFactor)
    for i in range (0, n): 
        for j in range(0, n): 
            result[i][j] = calculatePixelValue(imageToTransform, \
                tMatrix, \
                    (i*scalingFactor) + scalingFactor //2, \
                        (j*scalingFactor) + scalingFactor // 2, \
                            scalingFactor)
    return result

def binarify(blockifiedImage, targetCoverage): 
    flattenedImage = reduce(lambda z, y :z + y, blockifiedImage) 
    flattenedImage.sort()
    tIndex = targetCoverage * len(flattenedImage) // 100
    pivot = flattenedImage[tIndex]
    binarifiedImage = []
    for i in range(0, len(blockifiedImage)):
        binarifiedImage.append(list(map(lambda x: 0 if x > pivot else 1, blockifiedImage[i])))
    return binarifiedImage

window_name = "HI_AMY!"
imageName = './test.png'
img_codec = cv.IMREAD_COLOR
srcImage = cv.imread(cv.samples.findFile(imageName), img_codec)
print(srcImage.shape, flush=True)
block = blockify(srcImage, 50)
dst = binarify(block, 30)
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
cv.imshow(window_name, dst)
