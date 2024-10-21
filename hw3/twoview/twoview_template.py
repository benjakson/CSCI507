import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_keypoint(left_img, right_img):
    # find SIFT keypoints
    return key_points1, descriptor1, key_points2, descriptor2


def match_keypoints(descriptor1, descriptor2):
    # match SIFT descriptors
    return good_matches


if __name__ == "__main__":
    img1 = cv2.imread('data/left.jpg')
    img2 = cv2.imread('data/right.jpg')
    assert (img1 is not None) and (img2 is not None), 'cannot read given images'
    
    # camera intrinsic matrix
    f, cx, cy = 1000, 1024, 768 
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    # extract SIFT keypoints
    key_points1, descriptor1, key_points2, descriptor2 = get_keypoint(img1, img2)
    
    # match SIFT descriptors
    good_matches = match_keypoints(descriptor1, descriptor2)

    # calculate fundamental matrix
    pts1 = ...# from good_matche
    pts2 = ...# from good_matche
    F, inlier_mask = cv2.findFundamentalMat(...)
    print(f'* F = {F}')
    print(f'* number of inliers = {sum(inlier_mask.ravel())}')

    # show matched inlier features
    img_matched = cv2.drawMatches(img1, key_points1, img2, key_points2, good_matches, None, None, None,
                                matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
    cv2.namedWindow('Fundamental Matrix Estimation', cv2.WINDOW_NORMAL)
    cv2.imshow('Fundamental Matrix Estimation', img_matched)
    cv2.waitKey(0)
    
    # compute relative camera pose 
    E = K.T @ F @ K # essential matrix
    positive_num, R, t, positive_mask = cv2.recoverPose(...)
    print(f'* R = {R}')
    print(f'* t = {t}')

    # reconstruct 3D points (triangulation)
    P0 = K @ np.eye(3, 4, dtype=np.float32)
    Rt = np.hstack((R, t))
    P1 = K @ Rt
    pts1_inlier = pts1[] # select indliers
    pts2_inlier = pts2[] # select indliers
    X = cv2.triangulatePoints(P0, P1, pts1_inlier, pts2_inlier)
    X /= X[3]
    X = X.T

    # visualize 3D points
    ax = plt.figure(layout='tight').add_subplot(projection='3d')
    ax.plot(X[:,0], X[:,1], X[:,2], 'ro')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.grid(True)
    plt.show()

    