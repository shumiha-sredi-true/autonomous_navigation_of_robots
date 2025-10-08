import cv2
import numpy as np

img1 = cv2.imread('scale_begin.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('scale_end.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img1_orb = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_orb = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Key_points_1", img1_orb)
cv2.imshow("Key_points_2", img2_orb)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)


# Optionally take top-K or threshold by distance:
matches_cross = [m for m in matches if m.distance < 50]  # порог подбирается

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches_cross, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matches_check_cross", matched_img)

# Prepare points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])


if len(pts1) >= 4:
    H, mask = cv2.findHomography(
        pts1.reshape(-1,1,2),
        pts2.reshape(-1,1,2),
        cv2.RANSAC, 3.0
    )
    # фильтруем совпадения по маске RANSAC
    matchesMask = mask.ravel().tolist()
    inlier_matches = [m for i, m in enumerate(matches) if matchesMask[i]]
    print("Matches:", len(matches), "Inliers after RANSAC:", len(inlier_matches))

    matched_img_2 = cv2.drawMatches(
        img1, kp1, img2, kp2,
        inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Matches_2", matched_img_2)

cv2.waitKey(0)
cv2.destroyAllWindows()