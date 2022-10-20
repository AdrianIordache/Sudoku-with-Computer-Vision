import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#outliers = ['train/classic/26.jpg', 'train/classic/32.jpg', 'train/classic/36.jpg', 'train/classic/38.jpg', 'train/classic/5.jpg']

print("All Modules Imported")

class Line:
    """
    Store a line based on the two points.
    """
    def __init__(self, point_1, point_2):
        self.point_1 = point_1
        self.point_2 = point_2
        

class Point:
    def __init__(self, x, y):
        self.x = np.int32(np.round(x))
        self.y = np.int32(np.round(y))
    
    def get_point_as_tuple(self):
        return (self.x, self.y)
    

def compute_mean_and_std(image_paths, display = False):
    
    largest = []

    for (step, path) in enumerate(image_paths):
        #if path in outliers: continue
        #print(path)
        image = cv.imread(path)
        image = cv.resize(image, None, fx = 0.2, fy = 0.2)
        gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(gray, 100, 200)

        ret, thresh = cv.threshold(edges,10,255,0)
        contours, hierarchy = cv.findContours(edges, 1, 2)

        max_area, max_idx = 0, 0
        # for (idx, cnt) in enumerate(contours):
        #     area = cv.contourArea(cnt)      
        #     if area > max_area:
        #         max_area = area
        #         max_idx  = idx

        for (idx, cnt) in enumerate(contours):
            area = cv.contourArea(cnt)   
            perimeter = cv.arcLength(cnt, True)
            corners = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
            if area > max_area and len(corners) == 4:
                max_area = area

    largest.append(max_area)

    if display:
        print("Mean of the largest area extracted: {}".format(np.mean(largest)))
        print("STD of the largest area extracted: {}".format(np.std(largest)))
        plt.hist(largest, bins = 100)
        plt.show()

    return np.mean(largest), np.std(largest)

def generate_crop(contorurs, mean, std, n_std):

    best_fit_contour, best_fit_idx = np.inf, 0
    threshold = (mean + (std * n_std))

    for (idx, cnt) in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)    

        computed_area = w * h
        if np.abs(threshold - computed_area) < best_fit_contour:
            best_fit_contour = np.abs(threshold - computed_area)
            best_fit_idx     = idx

    return best_fit_idx


def compute_hough_transform(image, edges, threshold):
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold)

    vertical_lines: [Line] = []
    horizontal_lines: [Line] = [] 
    
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        angle = theta * 180 / np.pi
        # print(angle)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))  

        pt1 = (x1, y1) # x, y
        pt2 = (x2, y2) # x, y

        #cv.line(image, (x1,y1),(x2,y2),(0,0,255),2)

        if  1.4 <= theta <= 1.65:
            line = Line(Point(x=pt1[0], y=pt2[1]), Point(x=pt2[0], y=pt2[1])) 
            horizontal_lines.append(line)
        else:
            line = Line(Point(x=pt2[0], y=pt1[1]), Point(x=pt2[0], y=pt2[1])) 
            vertical_lines.append(line) 

    return horizontal_lines, vertical_lines
        
def get_rotation_angle(image, edges, threshold):
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold)

    angles = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        angle = theta * 180 / np.pi
        #print(angle)
        if angle < 90 + 30 and angle > 90 - 30:
            angles.append(angle)

    return angles

def remove_close_lines(lines: [Line], threshold: int, is_vertical: bool):
    
    different_lines = [] 
    if is_vertical:
        lines.sort(key=lambda line: line.point_1.x)
    else:
        lines.sort(key=lambda line: line.point_1.y)
    
    #  add the first line
    different_lines.append(lines[0])
    if is_vertical:
        for line_idx in range(1, len(lines)):
            if lines[line_idx].point_1.x - different_lines[-1].point_1.x > threshold:
                different_lines.append(lines[line_idx])
    else:
        for line_idx in range(1, len(lines)): 
            if lines[line_idx].point_1.y - different_lines[-1].point_1.y > threshold:
                different_lines.append(lines[line_idx])
    return different_lines
            
def generate_best_crop(contours, mean, std, n_std):
    best_fit_contour, best_fit_idx = np.inf, 0
    threshold = (mean + (std * n_std))

    for (idx, cnt) in enumerate(contours):
        computed_area = cv.contourArea(cnt)   
        perimeter = cv.arcLength(cnt, True)
        corners = cv.approxPolyDP(cnt, 0.02 * perimeter, True)

        if np.abs(threshold - computed_area) < best_fit_contour and len(corners) == 4:
            best_fit_contour = np.abs(threshold - computed_area)
            best_fit_idx     = idx

    return best_fit_idx

def generate_biggest_crop(contours):
    best_fit_contour, best_fit_idx = np.inf, 0

    for (idx, cnt) in enumerate(contours):
        computed_area = cv.contourArea(cnt)   
        perimeter = cv.arcLength(cnt, True)
        corners = cv.approxPolyDP(cnt, 0.02 * perimeter, True)

        if computed_area > best_fit_contour and len(corners) == 4:
            best_fit_contour = computed_area
            best_fit_idx     = idx

    return best_fit_idx

from PIL import Image
SCOPE = "classic"
PATH_TO_TRAIN  = "train/"
PATH_TO_IMAGES = os.path.join(PATH_TO_TRAIN, SCOPE)

image_paths = sorted(glob.glob(PATH_TO_IMAGES + "/*.jpg"))
label_paths = sorted(glob.glob(PATH_TO_IMAGES + "/*.txt"))

#mean, std = compute_mean_and_std(image_paths, display = True)

def reorder_points(corner):
    corner = corners.reshape(-1, 2)
    sorted_idx = np.zeros(4).astype(np.int64)

    the_sum = np.sum(corner, axis = 1)
    sorted_idx[0] = np.argmin(the_sum)
    sorted_idx[3] = np.argmax(the_sum)

    the_diff = np.diff(corner, axis = 1)
    sorted_idx[1] = np.argmin(the_diff)
    sorted_idx[2] = np.argmax(the_diff)

    return corner[sorted_idx]

mean, std = 230430.0, 0


for (step, path) in enumerate(image_paths):
    print(path)
    #path = "train/classic/32.jpg"
    #path = "train/classic/5.jpg"
    #path = "train/classic/26.jpg"

    image = cv.imread(path)
    image = cv.resize(image, None, fx = 0.2, fy = 0.2)
    gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 3)

    # ret, thresh = cv.threshold(gray,127,255,0)
    # cv.imshow("edges", thresh)
    # cv.waitKey(0)

    kernel = np.ones((7, 7), np.uint8)
    dilate = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv.erode(dilate, kernel, iterations = 3)

    edges = cv.Canny(dilate, 150, 250)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.imshow("edges", edges)
    cv.waitKey(0)
 
    idx = generate_crop(contours, mean, std, 0.5)
    best_idx = generate_biggest_crop(contours)
    print("Idx ", idx)
    print("Best Idx ", idx)

    cnt = contours[idx]
    x, y, w, h =  cv.boundingRect(cnt)

    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    # image = cv.rectangle(image,(x, y),(x + w, y + h),(0,255,0),2)
    # image = image[y_min : y_max, x_min : x_max]

    # cv.imshow("image", image)
    # cv.waitKey(0)

    frame = image.copy()

    # th = 0
    # height, width, _ = image.shape

    # x_min = max(0, x_min - th)
    # y_min = max(0, y_min - th)
    # x_max = min(width, x_max + th)
    # y_max = min(height, y_max + th)


    #image = cv.rectangle(image,(x, y),(x + w, y + h),(0,255,0),2)
    # cv.drawContours(frame, contours, -1,  (0, 255, 0), 3, cv.LINE_AA)
    # cv.imshow("edges", frame)
    # cv.waitKey(0)

    perimeter = cv.arcLength(cnt, True)
    corners = cv.approxPolyDP(cnt, 0.01 * perimeter, True)
    print(corners)
    corners = reorder_points(corners)

    # x_min = corners[0][0]
    # y_min = corners[0][1]
    # x_max = corners[3][0]
    # y_max = corners[3][1]

    # image = image[y_min : y_max, x_min : x_max]
    new_h, new_w = 500, 500
    warp_start = corners.astype(np.float32)
    warp_end   = np.array([
        [0, 0],
        [new_w, 0],
        [0, new_h],
        [new_w, new_h]]).astype(np.float32)

    warp_matrix = cv.getPerspectiveTransform(warp_start, warp_end)
    warp = cv.warpPerspective(image, warp_matrix, dsize = (new_w, new_h)) 

    cv.imshow("selected", warp)
    cv.waitKey(0)
    # break

    # # cv.drawContours(frame, contours, -1, (0, 255, 0), 3, cv.LINE_AA)
    # # cv.imshow("more", frame)
    # # cv.waitKey(0)

 
    # gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 100, 220)

    # angles = get_rotation_angle(image, edges, 100)

    # mean_angle = np.mean(angles)
    # rotation_angle = 90 - mean_angle

    # pil_image = Image.fromarray(image)
    # pil_image = pil_image.rotate((-1) * np.round(90 - mean_angle))
    # image = np.asarray(pil_image)

    # gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # edges = cv.Canny(gray, 100, 200)

    # ret, thresh = cv.threshold(edges,127,255,0)
    # contours, hierarchy = cv.findContours(thresh, 1, 2)
    
    # idx = generate_crop(contours, mean, std, 0.5)

    # cnt = contours[idx]
    # x, y, w, h =  cv.boundingRect(cnt)

    # x_min, y_min = x, y
    # x_max, y_max = x + w, y + h

    # image = cv.rectangle(image,(x, y),(x + w, y + h),(0,255,0),2)

    # image = image[y_min : y_max, x_min : x_max]
    # gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, 200, 220)

    # # cv.imshow("image", edges)
    # # cv.waitKey(0)

    # hl, vl = compute_hough_transform(image, edges, 160)
    # #hl = remove_close_lines(hl, 30, False)
    # vl = remove_close_lines(vl, 30, True)

    # for line in vl[:20]: 
    #     cv.line(image, line.point_1.get_point_as_tuple(), line.point_2.get_point_as_tuple(), (255, 0, 0), 2, cv.LINE_AA)
        

    # cv.imshow("image", image)
    # cv.waitKey(0)

    # # print(len(hl))
    # # print(len(ha))
    # #mean_angle = np.mean(ha)
    # # print(hl[:5])

    # 