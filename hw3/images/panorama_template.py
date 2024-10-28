import cv2
import numpy as np
import random

def stitch(left_img, right_img):
    # extract SIFT keypoints
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(left_img, right_img)
    
    # match SIFT descriptors
    good_matches = match_keypoints(descriptor1, descriptor2)
    
    # find homography using ransac
    src_pts = ...# from good_matches
    dst_pts = ...# from good_matches
    ransac_reproj_threshold = 5.0  # Threshold in pixels
    confidence = 0.99              # Confidence level
    maxIters = 10000               # Maximum number of iterations for RANSAC
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold, maxIters=maxIters, confidence=confidence)

    # combine images
    rows1, cols1 = right_img.shape[:2]
    rows2, cols2 = left_img.shape[:2]

    points1 =  np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    points  =  np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    points2 =  cv2.perspectiveTransform(points, homography_matrix)
    list_of_points = np.concatenate((points1,points2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    H_translation = (np.array([[1, 0, (-x_min)], [0, 1, (-y_min)], [0, 0, 1]])).dot(homography_matrix)

    output_img = cv2.warpPerspective(left_img, H_translation, (x_max-x_min, y_max-y_min))
    output_img[(-y_min):rows1+(-y_min), (-x_min):cols1+(-x_min)] = right_img
    result_img = output_img

    return result_img
    

def get_keypoint(left_img, right_img):
    # find SIFT keypoints
    return key_points1, descriptor1, key_points2, descriptor2


def match_keypoints(descriptor1, descriptor2):
    # match SIFT descriptors
    return good_matches


if __name__ == "__main__":
    # load all 8 images
    left_img = cv2.imread('data/field1.jpg')
    right_img = cv2.imread('data/field2.jpg')
    assert (left_img is not None) and (right_img is not None), 'cannot read given images'

    # downsample images
    height, width = left_img.shape[:2]
    left_img = cv2.resize(left_img, (width//5, height//5), interpolation=cv2.INTER_AREA)
    right_img = cv2.resize(right_img, (width//5, height//5), interpolation=cv2.INTER_AREA)

    result_img = stitch(left_img, right_img)

    cv2.imshow('Panorama Image', result_img)
    cv2.waitKey(0)

    cv2.imwrite('panorama.jpg', result_img)