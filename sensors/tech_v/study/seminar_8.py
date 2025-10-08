import cv2
import numpy as np

img1 = cv2.imread('photos/scale_begin.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('photos/scale_end.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img1_sift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Key_points_1", img1_sift)
cv2.imshow("Key_points_2", img2_sift)

# FLANN params for SIFT (float descriptors)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)

# Lowe's ratio test
ratio_thresh = 0.75
good = []
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good.append(m)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches", matched_img)

if len(good) >= 4:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    # mask — Nx1, 1=inlier, 0=outlier
    inlier_matches = [good[i] for i in range(len(good)) if mask[i][0]]
    print("Good matches:", len(good), "Inliers after RANSAC:", len(inlier_matches))
else:
    print("Not enough matches:", len(good))

matched_img_2 = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches_2", matched_img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()
