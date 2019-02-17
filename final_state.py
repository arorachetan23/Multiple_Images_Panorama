#updates     
#shift
#wrapping
#glob for all the images
#all images
#perspective
#better result
import sys
import argparse
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Warp img2 to img1 using the homography matrix H
def merge(img1, img2, H):
    r1, c1 = img1.shape[:2]
    r2, c2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,r1], [c1,r1], [c1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,r2], [c2,r2], [c2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    #print(x_min)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    #print(H)
    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    #center=[H[0][2],H[1][2]]
    output_img[translation_dist[1]:r1+translation_dist[1], translation_dist[0]:c1+translation_dist[0]] = img1

    #print(x_min,x_max)
    #return output_img,x_min,x_max,y_min,y_max
    return output_img

if __name__=='__main__':
    all_img = []
    for img in glob.glob("data/*.jpg"):
        image= cv2.imread(img)
        all_img.append(image)

    flag=1   
    fig, ax = plt.subplots()
    center1=[]
    min_match_count = 5

    for i in range(40):

        if flag==1:
            img1 = all_img[0]
            flag=0
        else :
            img1=result    


        img2=all_img[i+1]
        
   
        

        sift =cv2.xfeatures2d.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Initialize parameters for Flann based matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        # Initialize the Flann based matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        # Lowe's ratio test
        good_matches = []
        for m1,m2 in matches:
            if m1.distance < 0.7*m2.distance:
                good_matches.append(m1)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            #result,x_min,x_max,y_min,y_max = warpImages(img2, img1, M)
            result = merge(img2, img1, H)


        else:
            print ("Not Enough matches")
    
        
    
    for i in range(1):
        img2=all_img[i]
        img1=result
        sift =cv2.xfeatures2d.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Initialize parameters for Flann based matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        # Initialize the Flann based matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        # Lowe's ratio test
        good_matches = []
        for m1,m2 in matches:
            if m1.distance < 0.7*m2.distance:
                good_matches.append(m1)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([ keypoints1[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            #result,x_min,x_max,y_min,y_max = warpImages(img2, img1, M)
            result = merge(img2, img1, H)


        else:
            print ("Not Enough matches")

    #plt.show()      
    result=cv2.resize(result,None,fx=0.1,fy=0.1)
    cv2.imshow("image",result)
    cv2.imwrite('Fianl_map.png',result)
    cv2.waitKey()