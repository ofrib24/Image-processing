# ------------------------------ imports -----------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import map_coordinates
import shutil
from imageio import imwrite
from sol4_utils import *
import matplotlib.image as im
import skimage.color as ski
import scipy.signal as signal


# ------------------------------ constants ---------------------------------
KERNEL_SIZE = 3
K_COEFF = 0.04
N = 7
M = 7
RADIUS_SIZE = 3
DESC_RADIUS = 3
HOMOGRAPHY_SIZE = 3


# ------------------------------ functions ---------------------------------
def harris_corner_detector(image):
    """
    Harris corner detection algorithm, Finds features of an image.
    :param image: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    Ix = convolution_derivative(image, True)
    Iy = convolution_derivative(image, False)
    blurerd_squared_Ix = spatial_blur(np.power(Ix, 2), KERNEL_SIZE)
    blured_squared_Iy = spatial_blur(np.power(Iy, 2), KERNEL_SIZE)
    blured_IxIy = spatial_blur(Ix * Iy, KERNEL_SIZE)
    det_M = (blurerd_squared_Ix * blured_squared_Iy) - np.power(blured_IxIy, 2)
    trace_M = blurerd_squared_Ix + blured_squared_Iy
    R_H = det_M - (K_COEFF*np.power(trace_M, 2))
    maximas = non_maximum_suppression(R_H)
    return np.flip(np.argwhere(maximas))


def sample_descriptor(image, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param image: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    N = pos.shape[0]
    K = 1 + 2 * desc_rad
    descriptors_vec = np.zeros((N, K, K))
    for i in range(N):
        x = np.arange(pos[i, 0] - desc_rad, pos[i, 0] + (desc_rad + 1))
        y = np.arange(pos[i, 1] - desc_rad, pos[i, 1] + (desc_rad + 1))
        x_axis, y_axis = np.meshgrid(x, y)
        descriptor = map_coordinates(image, [y_axis, x_axis], order=1, prefilter=False)
        denomerator = np.linalg.norm(descriptor-np.mean(descriptor))
        if not denomerator:
            # avoiding division by zero
            denomerator = 1
        descriptors_vec[i, :, :] = ((descriptor - np.mean(descriptor)) / denomerator).reshape((K, K))
    return descriptors_vec


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    key_points = spread_out_corners(pyr[0], N, M, RADIUS_SIZE)
    feature_descriptor = sample_descriptor(pyr[2], 0.25*key_points, DESC_RADIUS)
    return [key_points, feature_descriptor]


def second_best_max(vec):
    """
    Finds the 2nd maximum of a given array and returns all the elements that are
    greater or equal to it.
    """
    clone = np.copy(vec)
    np.delete(clone, np.max(clone))
    return np.where(np.max(clone) <= vec, vec, 0)


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    N1 = desc1.shape[0]
    N2 = desc2.shape[0]
    K = desc1.shape[1]
    flatten1 = np.reshape(desc1, (N1, np.power(K, 2)))
    flatten2 = np.reshape(desc2, (N2, np.power(K, 2)))
    scores = np.dot(flatten1, flatten2.T)
    scores = np.apply_along_axis(second_best_max, 1, scores)
    final_scores = np.apply_along_axis(second_best_max, 1, scores.T).T
    desc1_matches, desc2_matches = np.where(min_score < final_scores)
    return [desc1_matches, desc2_matches]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12
    """
    arr = np.ones((pos1.shape[0], H12.shape[0]))
    arr[:, :2] = pos1
    xyz2_tilde = np.dot(H12, arr.T).T
    return xyz2_tilde[:, :2] / xyz2_tilde[:, 2:]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC method.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    max_score = 0
    curr_score = 0
    homography_matrix = np.zeros((HOMOGRAPHY_SIZE, HOMOGRAPHY_SIZE))
    return_array = None
    for i in range(num_iter):

        # 1 - sample
        x, y = np.random.choice(points1.shape[0], size=2, replace=False)

        # 2 - compute the line - linear equation
        H_12 = estimate_rigid_transform(np.array([points1[x], points1[y]]), np.array([points2[x], points2[y]]), translation_only)

        # 3 - score the line
        line_score = apply_homography(points1, H_12)
        squared_euclidean_distance = np.power(np.linalg.norm(line_score - points2, axis=1), 2)

        inliner_matches = np.argwhere(squared_euclidean_distance < inlier_tol)

        # decide if current score is maximal
        curr_score = len(inliner_matches)
        if max_score < curr_score:
            homography_matrix = H_12
            max_score = curr_score
            return_array = inliner_matches

    return [homography_matrix, return_array.reshape(return_array.shape[0])]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    points2[:, 0] = points2[:, 0] + im1.shape[1]
    outliers = np.setdiff1d(np.arange(points1.shape[0]), inliers)
    for i in range(points1.shape[0]):
        im1_matches_rows = points1[i, 0]
        im1_matches_cols = points1[i, 1]
        im2_matches_rows = points2[i, 0]
        im2_matches_cols = points2[i, 1]
        x = np.hstack((im1_matches_rows, im2_matches_rows))
        y = np.hstack((im1_matches_cols, im2_matches_cols))
        if i in inliers:
            plt.plot(x, y, mfc='r', c='y', lw=.3, ms=3, marker='.')
            plt.plot(im1_matches_rows, im1_matches_cols, color='red', marker='o', ms=1)
            plt.plot(im2_matches_rows, im2_matches_cols, color='red', marker='o', ms=1)
        if i in outliers:
            plt.plot(x, y, mfc='r', c='b', lw=.3, ms=3, marker='.')
            plt.plot(im1_matches_rows, im1_matches_cols, color='red', marker='o', ms=.8)
            plt.plot(im2_matches_rows, im2_matches_cols, color='red', marker='o', ms=.8)

    plt.imshow(np.hstack((im1, im2)), cmap="gray")
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    relative_homographies = [0] * (len(H_succesive) + 1)
    curr_homography = np.eye(3)
    for i in range(m, len(H_succesive)):
        if i == m:
            relative_homographies[m] = np.eye(3)
        curr_homography = curr_homography @ np.linalg.inv(H_succesive[i])
        relative_homographies[i+1] = curr_homography / curr_homography[2, 2]
        if i == len(H_succesive)-1:
            curr_homography = np.eye(3)
    i = m-1
    while 0 <= i:
        curr_homography = curr_homography @ H_succesive[i]
        relative_homographies[i] = curr_homography / curr_homography[2, 2]
        i -= 1
    return relative_homographies


def compute_bounding_box(homography, w, h):
    """
    Computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    new_points = apply_homography(np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]), homography)
    return np.array([[np.min(new_points[:, 0]), np.min(new_points[:, 1])],
                     [np.max(new_points[:, 0]), np.max(new_points[:, 1])]]).astype(np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    bounding_box = compute_bounding_box(homography, image.shape[1], image.shape[0])
    x, y = np.meshgrid(np.arange(bounding_box[0][0], bounding_box[1][0]),
                       np.arange(bounding_box[0][1], bounding_box[1][1]))
    coordinates = np.stack((x.flat, y.flat)).T
    indices = apply_homography(coordinates, np.linalg.inv(homography))
    rows = indices[:, 0].reshape(x.shape)
    cols = indices[:, 1].reshape(y.shape)
    return map_coordinates(image, [cols, rows], order=1, prefilter=False)


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret
