import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print("All Modules Imported")

SCOPE = "classic"
PATH_TO_TRAIN  = "train/"
PATH_TO_IMAGES = os.path.join(PATH_TO_TRAIN, SCOPE)

image_paths = sorted(glob.glob(PATH_TO_IMAGES + "/*.jpg"))
label_paths = sorted(glob.glob(PATH_TO_IMAGES + "/*.txt"))

for (step, path) in enumerate(image_paths):
    print(path)
    #path = "train/classic/32.jpg"
    #path = "train/classic/14.jpg"

    image = cv.imread(path)
    image = cv.resize(image, None, fx = 0.2, fy = 0.2)
    gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 100, 200)

    ret, thresh = cv.threshold(edges,10,255,0)
    contours, hierarchy = cv.findContours(edges, 1, 2)

    # max_area, max_idx = 0, 0
    # for (idx, cnt) in enumerate(contours):
    #     area = cv.contourArea(cnt)      
    #     if area > max_area:
    #         max_area = area
    #         max_idx  = idx

    # cnt = contours[max_idx]
    cnt = max(contours, key = cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    # image = cv.rectangle(image,(x, y),(x + w, y + h), (0, 255, 0), 2)
    # cv.imshow("image", image)
    # cv.waitKey(0)

    rect = cv.minAreaRect(cnt)
    #print(rect)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv.warpPerspective(image, M, (width, height))
    cv.imshow("image", warped)
    cv.waitKey(0)


    #for cnt in contours:
        #area = cv.contourArea(cnt)
        #x, y, w, h =  cv.boundingRect(cnt)

        # area = w * h
        # if area > max_area:
        #   max_area = area
        #   X, Y, W, H = x, y, w, h

        #cv.drawContours(image,[box],0,(0,0,255),2)

        # x, y, w, h =  cv.boundingRect(max_box)

    #image = cv.rectangle(image,(x1, y1),(x2, y2),(0,255,0),2)
    # cv.imshow("image", image)
    # cv.waitKey(0)