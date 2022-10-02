import math
import numpy as np
import cv2


def imgToMatrix(image):
    """
    Method to convert an image to its matrix representation
    :param image: File path of the image to convert
    :return: matrix representation of the image
    """
    imgMatrix = cv2.imread(image, 0)  # reading in grayscale
    return imgMatrix


def gaussianMask(sigma):
    """
    Method to generate a 1D gaussian filter (mask)
    :param sigma: The standard deviation for the gaussian function
    :return: A numpy array as a gaussian filter mask
    """
    filterSize = 3  # Filter size is fixed to be 3
    x = np.array(list(range(-int(filterSize / 2), int(filterSize / 2 + 1))))
    G_x = 1 / (math.sqrt(2 * math.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))  # Using the formula for Gaussian
    Gx = G_x.reshape((-1, filterSize))  # 1x3 filter
    Gy = Gx.reshape((filterSize, -1))  # 3x1 filter
    return Gx, Gy


def derivativeGaussian(Gx, Gy):
    """
    Method to obtain derivative of a gaussian filter
    :param Gx: X directional component of a gaussian filter
    :param Gy: Y directional component of a gaussian filter
    :return: Numpy arrays as directional derivatives of gaussian filter
    """
    x = np.array([[-1], [0], [1]])  # Backward derivative in X direction
    y = np.array([[-1, 0, 1]])  # Backward derivative in Y direction

    # obtaining the derivative mask
    dx = np.matmul(Gx.T, x.T)
    dy = np.matmul(y.T, Gy.T)
    return dx, dy


def padImage(G, I):
    """
    Method to add padding to an image
    :param G: Gaussian filter (kernel)
    :param I: Matrix of an image to pad
    :return: padding size and matrix representation of the padded image
    """
    filterSize, _ = G.shape
    padding = (filterSize - 1) // 2  # size of padding
    # padding the image on all sides by mirroring the edge pixels
    padded = np.pad(array=I, pad_width=padding, mode="edge")
    return padding, padded


def correlation(G, I, axis):
    """
    Method to perform element wise multiplication in the direction specified
    :param G: Gaussian filter (kernel)
    :param I: The image matrix
    :param axis: The axis (direction) along which the convolution should be performed
    :return: The resultant numpy array of the convolution of kernel with the image in the specified direction
    """
    G = np.rot90(G, k=2)  # flip the filter twice for convolution
    padding, paddedImage = padImage(G, I)
    rowsP, columnsP = paddedImage.shape
    dI = np.zeros_like(I)  # initialize the output
    if axis == "x":  # for X direction
        for row in range(padding, rowsP - padding):
            for col in range(padding, columnsP - padding):
                dI[row - padding, col - padding] = np.sum(
                    np.multiply(paddedImage[row - padding:row + padding + 1, col - padding:col + padding + 1], G))
    elif axis == "y":  # for Y direction
        for col in range(padding, columnsP - padding):
            for row in range(padding, rowsP - padding):
                dI[row - padding, col - padding] = np.sum(
                    np.multiply(paddedImage[row - padding:row + padding + 1, col - padding:col + padding + 1], G))
    return dI


def convolution(Ix, Iy, Gx, Gy):
    """
    A method to perform the convolution
    :param Ix: The image matrix along the rows
    :param Iy: The image matrix along the columns
    :param Gx: The Gaussian filter along the rows
    :param Gy: The Gaussian filter along the columns
    :return: The resultant numpy arrays of the convolution of kernel with the image
    """
    dIx = correlation(Gx, Ix, "x")
    dIy = correlation(Gy, Iy, "y")
    return dIx, dIy


def getMagnitude(Ix, Iy):
    """
    Method to compute the magnitude(gradient) of the convolved image along the X and Y direction
    :param Ix: The convoluted image along X direction
    :param Iy: The convoluted image along Y direction
    :return: The resultant magnitude
    """
    return np.sqrt(np.square(Ix) + np.square(Iy))


def getDirection(Ix, Iy):
    """
    Method to compute the gradient direction of the convolved image
    :param Ix: The convoluted image along X direction
    :param Iy: The convoluted image along Y direction
    :return: The resultant direction
    """
    return np.arctan2(Iy, Ix)


def nonMaximumSuppression(mags, dirs):
    """
    Method to implement non-maximum suppression on the pixels
    :param mags: Magnitudes of the convoluted image
    :param dirs: Directions of the convoluted image
    :return: Numpy array representing the suppressed image
    """
    dirs = dirs * 180 / math.pi
    dirs[dirs < 0] += 180
    output = np.zeros_like(mags)

    for x in range(1, mags.shape[0] - 1):
        for y in range(1, mags.shape[1] - 1):
            # thresholding the direction to either of the 4 directions viz 0 (or 180), 45, 90 and 135
            if (0 <= dirs[x, y] < 22.5) or (157.5 <= dirs[x, y] <= 180):
                upperPixel = mags[x, y + 1]
                lowerPixel = mags[x, y - 1]
            elif 22.5 <= dirs[x, y] < 67.5:
                upperPixel = mags[x + 1, y - 1]
                lowerPixel = mags[x - 1, y + 1]
            elif 67.5 <= dirs[x, y] < 112.5:
                upperPixel = mags[x + 1, y]
                lowerPixel = mags[x - 1, y]
            elif 112.5 <= dirs[x, y] < 157.5:
                upperPixel = mags[x - 1, y - 1]
                lowerPixel = mags[x + 1, y + 1]
            else:
                upperPixel = 0
                lowerPixel = 0

            # retain the value of pixel if it is between the thresholds
            if (mags[x, y] >= upperPixel) and (mags[x, y] >= lowerPixel):
                output[x, y] = mags[x, y]
            else:
                output[x, y] = 0
    return output


def exploreEdges(image, i, j, visited):
    """
    Method to find connected edges, used in hysteresis thresholding
    Implements a breadth-first search algorithm to find connected edges
    :param image: Matrix representation of image
    :param i: row index
    :param j: column index
    :param visited: Matrix to keep a track of visited edges
    :return: A boolean indicating whether the edge is connected or not
    """
    if (i > 0 and j > 0) and (i < image.shape[0] and j < image.shape[1] and visited[i, j] == 0):
        visited[i, j] = 1  # denoted current pixel as visited
        if image[i, j] == 0:  # check if it's an edge
            return False
        elif image[i, j] == 255:  # return true for edge
            return True
        else:  # check for neighbouring pixels - depth first approach
            return (exploreEdges(image, i - 1, j, visited) or exploreEdges(image, i + 1, j, visited) or
                    exploreEdges(image, i, j - 1, visited) or exploreEdges(image, i, j + 1, visited) or
                    exploreEdges(image, i - 1, j - 1, visited) or exploreEdges(image, i - 1, j + 1, visited) or
                    exploreEdges(image, i + 1, j - 1, visited) or exploreEdges(image, i + 1, j + 1, visited))
    return False


def hysteresisThresholding(image, lowerLimit, higherLimit):
    """
    Method to apply hysteresis thresholding on an image
    :param image: Numpy array representing the image after NMS
    :param lowerLimit: lower threshold limit
    :param higherLimit: higher threshold limit
    :return: Numpy array representing the image after thresholding
    """
    output = np.zeros_like(image)
    rows, cols = output.shape

    for i in range(rows):
        for j in range(cols):
            # check for pixel thresholds and assign its values to either white or black
            # white will indicate that the pixel belongs to an edge
            if image[i, j] > higherLimit:
                output[i, j] = 255  # set the pixel value to White if it's above the upper threshold value
            elif image[i, j] < lowerLimit:
                output[i, j] = 0  # set the pixel value to Black if it's below the lower threshold value
            elif lowerLimit <= image[i, j] <= higherLimit:
                # for pixels between the threshold limits, explore its neighbours and determine if the pixel
                # belongs to an edge or not. the exploration is done via depth first search
                visited = np.zeros_like(output)
                if exploreEdges(output, i, j, visited):  # if method returns True, pixel belongs to an edge
                    output[i, j] = 255
                else:   # otherwise, it doesn't. color it black
                    output[i, j] = 0

    return output


if __name__ == '__main__':

    """
    Main method that invokes all the methods to perform the canny edge detection
    """

    img = input("Enter the absolute path to the image (in double quotes): ")
    img = img[1:-1]
    I = imgToMatrix(img)

    stddev = float(input("Enter the standard deviation for gaussian function: "))

    print("=======================================================")
    print("Creating a gaussian mask")
    Gx, Gy = gaussianMask(stddev)

    print("=======================================================")
    print("Computing derivative of the gaussian mask")
    dGx, dGy = derivativeGaussian(Gx, Gy)

    print("=======================================================")
    print("Convoluting image with the gaussian filter")
    Ix, Iy = convolution(I, I, Gx, Gy)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_Ix.jpg", Ix)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_Iy.jpg", Iy)

    print("=======================================================")
    print("Convoluting image with the gaussian derivative")
    dIx, dIy = convolution(Ix, Iy, dGx, dGy)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_dIx.jpg", dIx)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_dIy.jpg", dIy)

    print("=======================================================")
    print("Computing magnitude of the image")
    magnitude = getMagnitude(dIx, dIy)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_Magnitude.jpg", magnitude)

    print("=======================================================")
    print("Computing direction of the image")
    direction = getDirection(dIx, dIy)

    print("=======================================================")
    print("Applying Non-Maximum Suppression")
    nms = nonMaximumSuppression(magnitude, direction)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_NMS.jpg", nms)

    print("=======================================================")
    print("Applying hysteresis thresholding")
    hyThresholding = hysteresisThresholding(nms, 25, 55)
    cv2.imwrite(str(img)[:-5] + "_StdDev_" + str(stddev) + "_hyt.jpg", hyThresholding)

    print("=======================================================")
    print("DONE")
    print("=======================================================")
