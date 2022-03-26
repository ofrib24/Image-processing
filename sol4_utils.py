from scipy.signal import convolve2d
import numpy as np
import matplotlib.image as im
import skimage.color as ski
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import scipy.signal as signal
from scipy.ndimage import convolve


# ------------------------------ constants ---------------------------------
GRAYSCALE_REP = 1
MAX_GRAY = 255
DERIVATIVE_MATRIX = np.array([[1], [0], [-1]])


# -------------------------- functions -------------------------------------
def convolution_derivative(image, x):
    """
    Takes the derivative of a given image using convolution
    :param image: image to derive
    :param x: true if x derivative is wanted, false if y
    :return: the magnitude matrix of the x and y derivatives of the image
    """
    if x:
        return convolve(image, DERIVATIVE_MATRIX)

    return convolve(image, DERIVATIVE_MATRIX.T)


def gaussian_kernel(kernel_size):
    """
    Creates a gaussian kernel sized kernel_size using convolution
    :param kernel_size:
    :return: gaussian kernel normalized
    """
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def spatial_blur(image, kernel_size):
    """
    Blurs an image using 2D convolution with gaussian kernel
    :param image:
    :param kernel_size:
    :return:
    """
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(image)
    if len(image.shape) == 2:
        blur_img = convolve2d(image, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(image[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    Reads an image and returns it with the given representation
    :param filename: the path to the image file
    :param representation: RGB or grayscale image representation
    :return: the new representation image
    """
    if representation == GRAYSCALE_REP:
        img = im.imread(filename) / MAX_GRAY
        return ski.rgb2gray(img[:, :, :3]).astype(np.float64)
    img = plt.imread(filename) / MAX_GRAY
    return img.astype(np.float64)


# -------------------------- Pyramid Blending -------------------------------------
def gaussian_filter_maker(filter_size):
    """
    Creates a gaussian filter approximation with binomial coefficients
    :param filter_size: number of coefficients
    :return: normalized binomial coefficients vector
    """
    array_for_conv = np.array([[1, 1]]).astype(np.float64)
    gaussian_filter = np.array([[1, 1]]).astype(np.float64)
    for i in range(filter_size-2):
        gaussian_filter = signal.convolve2d(gaussian_filter, array_for_conv)
    return gaussian_filter / np.sum(gaussian_filter)


def blur_image(image, blur_filter):
    """
    Smoothing an image using 1D convolution
    :param image:
    :param blur_filter:
    :return: blurred image
    """
    return filters.convolve(filters.convolve(image, blur_filter), blur_filter.T)


def sub_sample(image):
    """
    Samples every 2nd pixel in every 2nd row
    :param image:
    :return: new smaller image
    """
    return image[::2, ::2]


def reduce(image, image_filter):
    """
    Reduces the size of an image and using a filter to smooth
    and then sub-sampling every 2nd pixel
    :param image: an image
    :param image_filter: image filter
    :return: new smaller image
    """
    new_image = blur_image(image, image_filter)
    new_image = sub_sample(new_image)
    return new_image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds gaussian pyramid
    :param im: an image
    :max_levels: the maximum number of pyramid layers
    :filter_size: the size of the filter used for smoothing
    :return: gaussian pyramid and the filter used to build it
    """
    pyr = [im]
    gaussian_filter = gaussian_filter_maker(filter_size)
    max_levels = int(min(max_levels, np.log2(pyr[0].shape[0] // 16)+1, np.log2(pyr[0].shape[1] // 16)+1))
    for i in range(max_levels-1):
        pyr.append(reduce(pyr[i], gaussian_filter))
    return pyr, gaussian_filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds laplacian pyramid
    :param im: an image
    :param max_levels: the maximum number of pyramid layers
    :param filter_size: the size of the filter used for smoothing
    :return: laplacian pyramid and the filter used to build it
    """
    pyr, gaussian_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    new_pyr = []
    for i in range(len(pyr) - 1):
        new_pyr.append(pyr[i] - expand(pyr[i+1], gaussian_filter))
    new_pyr.append(pyr[-1])
    return new_pyr, gaussian_filter


def expand(image, image_filter):
    """
    Expands a given image by padding with zeros
    between each 2 pixels. we multiply the filter by 2 to maintain brightness
    :param image: an image
    :param image_filter: image filter
    :return: new larger image
    """
    new_image = np.zeros((image.shape[0]*2, image.shape[1]*2))
    new_image[::2, ::2] = image
    return blur_image(new_image, 2*image_filter)


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Takes a lplacian pyramid and makes an image from it.
    :lpyr: laplacian pyramid
    :filter_vec: filter vector for smoothing
    :coeff: coefficients vector
    :return: an image made from the pyramid
    """
    new_image = lpyr[-1] * coeff[-1]
    for i in range(2, len(lpyr)+1):
        new_image = lpyr[-i] + expand(new_image, filter_vec)
    return new_image


def render_pyramid(pyr, levels):
    """
    Creates an image that icludes every level of this image pyramid,
    gaussian or laplacian
    :param pyr: gaussian or laplacian pyramid
    :param levels: levels of pyramid to render
    :return: result image of the pyramid's layers
    """
    cols = 0
    stretched = []
    for i in range(levels):
        image = pyr[i]
        cols += image.shape[1]
        stretched.append(np.round(255 * (image - np.min(image)) / (np.max(image) - np.min(image))))
    res = np.zeros((pyr[0].shape[0], cols))
    start_index = 0
    for i in range(levels):
        image = stretched[i]
        res[:image.shape[0], start_index:start_index + image.shape[1]] = image
        start_index += image.shape[1]
    return res


def display_pyramid(pyr, levels):
    """
    Shows an image from a given pyramid
    :param pyr: gaussian or laplacian pyramid
    :param levels: number of pyramid levels
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Performs pyramid blending using gaussian and laplacian pyramids
    :param im1: first image
    :param im1: second image
    :param mask: mask
    :param max_levels: maximum levels in the pyramids
    :param filter_size_im:
    :param filter_size_mask:
    :return: the new blended image
    """
    new_image = np.zeros(im1.shape)
    mask = mask.astype(np.float64)
    if len(im1.shape) == 3:
        Gm, gaussian_filter3 = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
        for i in range(3):
            L1, gaussian_filter1 = build_laplacian_pyramid(im1[:, :, i], max_levels, filter_size_im)
            L2, gaussian_filter2 = build_laplacian_pyramid(im2[:, :, i], max_levels, filter_size_im)
            new_laplacian = Gm * np.array(L1) + (1 - np.array(Gm)) * L2
            coeff = np.ones((len(new_laplacian)))
            new_image[:, :, i] = laplacian_to_image(new_laplacian, gaussian_filter3, coeff)
        return np.clip(new_image, 0, 1)

    L1, gaussian_filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, gaussian_filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    Gm, gaussian_filter3 = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    new_laplacian = Gm * np.array(L1) + (1-np.array(Gm)) * L2
    coeff = np.ones((len(new_laplacian)))
    return np.clip(laplacian_to_image(new_laplacian, gaussian_filter3, coeff), 0, 1)

