#!/usr/bin/env python
# coding: utf-8

# ### Computer Vision - Laboratory class 2
# ### Automatic grading of multiple choice tests

# <img src="image.png" width=300 />

# In[13]:


import cv2 as cv 
import numpy as np
import math
import glob


# In[2]:


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
    
        
class Patch:
    """
    Store information about each item (where the mark should be found) in the table. 
    """
    def __init__(self, image_patch, x_min, y_min, x_max, y_max, line_idx, column_idx):
        self.image_patch = image_patch
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.line_idx = line_idx
        self.column_idx = column_idx
        self.has_x: int = 0 # 0 meaning it does not contain an 'X', 1 meaning it contains an 'X'
    
    def set_x(self, has_x: int):
        assert has_x == 0 or has_x == 1 # convention 
        self.has_x = has_x


# In[3]:


def show_image(image, window_name='image', timeout=0):
    """
    :param timeout. How many seconds to wait untill it close the window.
    """
    cv.imshow(window_name, cv.resize(image, None, fx=0.6, fy=0.6))
    cv.waitKey(timeout)
    cv.destroyAllWindows()
    
    
def draw_lines(image, lines: [Line], timeout: int = 0, color: tuple = (0, 0, 255),
               return_drawing: bool = False, window_name: str = 'window'):
    """
    Plots the lines into the image.
    :param image.
    :param lines.
    :param timeout. How many seconds to wait untill it close the window.
    :param color. The color used to draw the lines
    :param return_drawing. Use it if you want the drawing to be return instead of displayed.
    :return None if return_drawing is False, otherwise returns the drawing.
    """
    drawing = image.copy()
    if drawing.ndim == 2:
        drawing = cv.cvtColor(drawing, cv.COLOR_GRAY2BGR)
    for line in lines: 
        cv.line(drawing, line.point_1.get_point_as_tuple(), line.point_2.get_point_as_tuple(), color, 2, cv.LINE_AA)
        
    if return_drawing:
        return drawing
    else:
        show_image(drawing, window_name=window_name, timeout=timeout)

        
def show_patches_which_have_x(patches: [Patch]) -> None:
    """
    This function draws a colored rectangle if the patch has an 'X'. 
    """
 
    image_color = np.zeros((600, 400, 3), np.uint8) # it something crashed it is because of this dimension, it may be too small.
    x_min = patches[0].x_min
    y_min = patches[0].y_min
    for patch in patches:
        x_min_current = patch.x_min - x_min
        y_min_current = patch.y_min - y_min
        x_max_current = patch.x_max - x_min
        y_max_current = patch.y_max - y_min   
        image_color[y_min_current: y_max_current, x_min_current: x_max_current] = np.dstack((patch.image_patch, patch.image_patch, patch.image_patch))

        if patch.has_x == 1:  
            cv.rectangle(image_color, (x_min_current, y_min_current), 
                         (x_max_current, y_max_current), color=(255, 0, 0), thickness=1)
         
    show_image(image_color, window_name='patches', timeout=0)


# In[4]:


def remove_close_lines(lines: [Line], threshold: int, is_vertical: bool):
    """
    It removes the closest lines.
    :param lines.
    :param threshold. It specify when the lines are too close to each other.
    :param is_vertical. Set it to True or False.
    :return : The different lines.
    """
    
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
            


# In[5]:


def get_patches(lines: [Line], columns: [Line], image, show_patches: bool = False) -> [Patch]:
    """
    It cuts out each box from the table defined by the lines and columns.
    :param lines. The lines that difine the table.
    :param columns. The columns that difine the table.
    :param image. The image containing the table.
    :param show_patches. Determine if the patches will be drawn on the image or not.
    :return : A list with all boxes in the table.
    """
    
    def crop_patch(image_, x_min, y_min, x_max, y_max):
        """
        Crops the bounding box represented by the coordinates.
        """
        return image_[y_min: y_max, x_min: x_max].copy()
    
    def draw_patch(image_, patch: Patch, color: tuple = (255, 0, 255)):
        """
        Draw the bounding box corresponding to the patch on the image.
        """
        cv.rectangle(image_, (patch.x_min, patch.y_min), (patch.x_max, patch.y_max), color=color, thickness=5)
    
    assert image.ndim == 2
    if show_patches: 
        image_color = np.dstack((image, image, image))
  
    lines.sort(key=lambda line: line.point_1.y)
    columns.sort(key=lambda column: column.point_1.x)
    patches = []
    step = 5
    for line_idx in range(len(lines) - 1):
        for col_idx in range(len(columns) - 1):
            current_line = lines[line_idx]
            next_line = lines[line_idx + 1] 
            
            y_min = current_line.point_1.y + step
            y_max = next_line.point_1.y - step
            
            current_col = columns[col_idx]
            next_col = columns[col_idx + 1]
            x_min = current_col.point_1.x + step 
            x_max = next_col.point_1.x - step
            
            patch = Patch(image_patch=crop_patch(image,  x_min, y_min, x_max, y_max),
                          x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                          line_idx=line_idx, column_idx=col_idx)
            
            if show_patches:
                draw_patch(image_color, patch)
            
            patches.append(patch)
            
    if show_patches:
        show_image(image_color, window_name='patches', timeout=0)
    return patches


# In[6]:


class MagicClassifier:
    """
    A very strong classifier that detects if the patch has an 'X' or not.
    """
    def __init__(self):
        self.threshold = 245
    
    def classify(self, patch: Patch) -> int:
        """
        Receive a Patch and return 1 if there is an 'X' in the pacth and 0 otherwise.
        """ 
        if patch.image_patch.mean() > self.threshold:
            return 0
        else: 
            return 1
        
        
def classify_patches_with_magic_classifier(patches: [Patch]) -> None:
    """
    Receive the patches and classify if the patch contains an 'X' or not.
    :param patches.
    :return None
    """
    magic_classifier = MagicClassifier()
    for patch in patches:
        patch.set_x(magic_classifier.classify(patch))


# In[7]:


def transform_gt(gt_text): 
    """
    Transform the ground-truth from text to matrix. For example "1 C" meaning "on the line 1 the correct answer is C"
    will be tranform in "0 0 1 0".
    """
    num_lines = len(gt_text)
    char_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    gt = np.zeros((num_lines, 4), int)
    for gt_line in gt_text:
        line_idx = int(gt_line[0]) - 1
        if line_idx >= num_lines:
            line_idx -= num_lines # we do this because of the gt format 1 A; 2 B, ..., 15 B, 16 A, 30 C. 
            
        col_idx = char_to_index[gt_line[1]]
        gt[line_idx, col_idx] = 1
    return gt


def compute_accuracy(patches: [Patch], gt) -> float:
    """
    :param patches
    :param gt. The ground-truth in a matrix format: [[0 0 1 0], [1, 0, 0, 0], ..., [1, 0, 0, 0]]
    This function computes the accuracy of our approach of detecting the marked boxes in table.
    """
    num_correct_answers = 0 
    
    def compute_num_answers_per_line(patches_, num_lines):
        num_answers_per_line = np.zeros((num_lines))
        for patch in patches_:
            if patch.has_x == 1:
                num_answers_per_line[patch.line_idx] += 1
        return num_answers_per_line
        
         
    num_answers_per_line = compute_num_answers_per_line(patches, num_lines=len(gt))
    for patch in patches:
        line_idx = patch.line_idx
        col_idx = patch.column_idx
        if gt[line_idx, col_idx] == 1 and patch.has_x == 1 and num_answers_per_line[line_idx] == 1:
            num_correct_answers += 1
            
    accuracy = num_correct_answers / len(gt) 
    return accuracy


# In[8]:


def get_horizontal_and_vertical_lines_hough(edges, threshold: int = 160) -> tuple:
    """
    Returns the horizontal and vertical lines found by Hough transform.
    :param edges = the edges of the image.
    :threshold = it specifies how many votes need a line to be considered a line.
    :return (horizontal_lines: List(Line), vertical_lines: List(Line))
    """
    
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold)

    assert lines is not None
    
    vertical_lines: [Line] = []
    horizontal_lines: [Line] = [] 
    
    for i in range(0, len(lines)):
        # TODO: compute the line coordinate
        rho = lines[i][0][0]
        theta = lines[i][0][1]

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
        
        if  1.4 <= theta <= 1.65:
            line = Line(Point(x=pt1[0], y=pt2[1]), Point(x=pt2[0], y=pt2[1])) 
            horizontal_lines.append(line)
        else:
            line = Line(Point(x=pt2[0], y=pt1[1]), Point(x=pt2[0], y=pt2[1])) 
            vertical_lines.append(line) 
             
    return horizontal_lines, vertical_lines


# In[9]:


def find_patches_hough(table_image, is_left: bool, show_intermediate_results=False) -> [Patch]:
    """
    This function finds the 'X' in the table using the following steps: 
    1. split the image into the left or the right part
    2. convert the image to grayscale
    3. find the edges of the image using Canny (threshold1=100 and threshold2=150)
    4. find the horizontal and vertical lines using Hough Transform
    5. delete lines that are too close to each other (less than 30 pixels)
    6. keep only the last 5 verical lines and the last 16 horizontal lines
    7. cut out the paches (the boxes) from the table based on the horizontal and vertical lines 
    """
    
    # split the image into left and right parts
    if is_left: 
        table_image = table_image[int(0.5 * table_image.shape[0]): int(0.9 * table_image.shape[0]),
                                :int(0.5 * table_image.shape[1])]
    else:
        table_image = table_image[int(0.5 * table_image.shape[0]): int(0.9 * table_image.shape[0]),
                                int(0.5 * table_image.shape[1]):]
    
    
    gray_image = cv.cvtColor(table_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_image, 50, 100)
    if show_intermediate_results: 
        show_image(edges, window_name='edges', timeout=0)
        
    horizontal_lines, vertical_lines = get_horizontal_and_vertical_lines_hough(edges, 160)
    if show_intermediate_results:
        draw_lines(gray_image, horizontal_lines, window_name='Horizontal lines')
        draw_lines(gray_image, vertical_lines, window_name='Vertical lines')
    
    distinct_horizontal_lines = remove_close_lines(horizontal_lines, 30, False)
    distinct_vertical_lines = remove_close_lines(vertical_lines, 30, True)
    
    # take the last 5 verical lines and the last 16 horizontal lines
    distinct_horizontal_lines = distinct_horizontal_lines[-16:]
    distinct_vertical_lines = distinct_vertical_lines[-5:]
    
    if show_intermediate_results:
        draw_lines(gray_image, distinct_vertical_lines, window_name='Vertical lines after removing')
        draw_lines(gray_image, distinct_horizontal_lines, window_name='Horizontal lines after removing')
    
    patches = get_patches(distinct_horizontal_lines, distinct_vertical_lines, gray_image, 
                          show_patches=show_intermediate_results)
     
    return patches


# In[10]:


# define metrics
def mean_ssd(image_1, image_2) -> float:
    """
     This function receives two matrices having the same dimension and returns the mean of squared differences them.
    :param image_1. The first matrix
    :param image_2. The second matrix
    :return float. The mean of squared differences.
    """ 
    image_1 = np.float32(image_1)  
    image_2 = np.float32(image_2)  
    return np.mean((image_1 - image_2) ** 2)
 

def get_displacement_vector(img_query, img_template, window_i: tuple = (-15, 15), window_j: tuple = (-15, 15)) -> tuple: 
    """
    It returns (pos_i, pos_j) which is the best alignment of channel_1 with blue_channel.
    :param channel_1. This is the channel that will be alligned to the blue one.
    :param blue_channel
    :window_i:tuple. The start and end position on the y axis.
    :window_j:tuple. The start and end position on the x axis.
    :return (pos_i, pos_j) which is the best alignment of channel_1 with blue_channel
    """
    pos_i = 0; pos_j = 0
    min_error = np.inf
    height_1, width_1 = img_query.shape        
    height_blue, width_blue = img_template.shape
    
    for i in range(window_i[0], window_i[1]):
        for j in range(window_j[0], window_j[1]):   
            if i >= 0:
                ymin_blue = i
                ymax_blue = height_blue
                ymin_1 = 0
                ymax_1 = height_1 - i
            else:
                ymin_blue = 0
                ymax_blue = height_blue + i
                ymin_1 = -i
                ymax_1 = height_1
               
            if j >= 0:
                xmin_blue = j
                xmax_blue = width_blue
                xmin_1 = 0
                xmax_1 = width_1 - j
            else:
                xmin_blue = 0
                xmax_blue = width_blue + j
                xmin_1 = -j
                xmax_1 = width_1
                
            patch_1 = img_query[ymin_1: ymax_1, xmin_1: xmax_1]
            patch_2 = img_template[ymin_blue: ymax_blue, xmin_blue: xmax_blue]        
            assert patch_1.shape == patch_2.shape
                
            value = mean_ssd(patch_1, patch_2) 
            if min_error > value:
                min_error = value
                pos_i = i
                pos_j = j   
    return pos_i, pos_j

  
def allign_image_based_on_displacement_vector(image_, pos_i, pos_j):
    """
    Receive the image and the displacement vector and reconstruct the image according to the  displacement vector.
    """
    window_size = max(np.abs([pos_i, pos_j]))
    height, width, _ = image_.shape  
    allign_image = 255 * np.ones((height + window_size * 2, width + window_size * 2, 3), np.uint8) 
    allign_image[pos_i + window_size: pos_i + window_size + height,
                 pos_j + window_size: pos_j + window_size + width] = image_.copy()

    return allign_image
 

def find_patches_template(image, template, show_intermediate_results: bool = False):
    """
    This function finds the 'X' in the entire image using the following steps: 
    1. transforming the query image and the template to grayscale
    2. getting the edges for both images using Canny edge detector with threshold1=100 and threshold2=150
    3. obtaining the displacement vector between the query image and the template image
    4. allign the query image to the template
    5. obtain the first left/right corner of the first table and the second table
    6. getting the vertical/horizontal lines based on the table dimension and the number of vertical/horizontal lines based
    7. cut out the paches (the boxes) from the two tables based on the horizontal and vertical lines 
    """
    query_image_edges = ...
    template_image_edges = ...
    
    if show_intermediate_results: 
        show_image(query_image_edges, window_name='query_image_edges', timeout=0)
        show_image(template_image_edges, window_name='template_image_edges', timeout=0)
        
    pos_i, pos_j = get_displacement_vector(query_image_edges, template_image_edges)
    alligned_image = allign_image_based_on_displacement_vector(image, pos_i, pos_j)
    
    left_corner_1 = Point(x=183, y=658)
    right_corner_1 = Point(x=183 + 115, y=657 + 343)

    left_corner_2 = Point(x=620, y=658)
    right_corner_2 = Point(x=620 + 115, y=657 + 343)

    step_y = ...
    step_x = ...
    
    def get_vertical_and_horizontal_lines(left_corner: Point, step_x_, step_y_, window_size_): 
        vertical_lines = []
        horizontal_lines = []
        
        # TODO: compute the vertical/horizontal lines based left corner step and window_size 
        ...
        return vertical_lines, horizontal_lines
     
    
    window_size = np.abs([pos_i, pos_j]).max()
    vertical_lines_left, horizontal_lines_left = get_vertical_and_horizontal_lines(left_corner_1, step_x, step_y, 
                                                                                   window_size_=window_size) 
    
    alligned_image_gray = ...
    left_patches = get_patches(horizontal_lines_left, vertical_lines_left, alligned_image_gray, 
                               show_patches=show_intermediate_results)
 
    vertical_lines_right, horizontal_lines_right = get_vertical_and_horizontal_lines(left_corner_2, step_x, step_y,
                                                                                     window_size_=window_size) 
    right_patches = get_patches(horizontal_lines_right, vertical_lines_right, alligned_image_gray,
                               show_patches=show_intermediate_results)
    
    return left_patches, right_patches


# In[11]:


def solve_image(image_path: str, gt_path: str, use_hough: bool = True, show_intermediate_results: bool = False) -> float:
    """
    This function 'reads' the 'X' from the table and return the accuracy of detecting the correct position of 'X'.
    :param image_path. The path to the image that will be solved.
    :param gt_path. The path to the grount truth txt file.
    :param use_hough. If true hough transform will be used, otherwise template matching will be used.
    :param show_intermediate_results. Set it to true to see intermediate image results.
    :return. The accuracy of detecting the correct position of 'X'
    """
    
    original_image = cv.imread(image_path)
    assert original_image is not None  
    
    if use_hough:
        original_image = cv.resize(original_image, None, fx=0.3, fy=0.3)
    else:
        original_image = cv.resize(original_image, None, fx=0.2, fy=0.2)

    
    if use_hough:
        patches_left_image = find_patches_hough(original_image, is_left=True, 
                                                show_intermediate_results=show_intermediate_results)
        patches_right_image = find_patches_hough(original_image, is_left=False, 
                                                 show_intermediate_results=show_intermediate_results)
    else:
        template = cv.resize(cv.imread("image_template.jpg"), None, fx=0.2, fy=0.2) 
        patches_left_image, patches_right_image = find_patches_template(original_image, template, 
                                                                        show_intermediate_results=show_intermediate_results)
    classify_patches_with_magic_classifier(patches_left_image) 
    classify_patches_with_magic_classifier(patches_right_image) 
    
    if show_intermediate_results:  
        show_patches_which_have_x(patches_left_image)
        show_patches_which_have_x(patches_right_image) 
    
    def get_gt_left_right(gt_path_):
        ground_truth_content = np.loadtxt(gt_path_, dtype=str)
        ground_truth_left = ground_truth_content[1:16]
        ground_truth_right = ground_truth_content[16:-1] 
        gt_left, gt_right = [transform_gt(gt) for gt in [ground_truth_left, ground_truth_right]]
        return gt_left, gt_right
    
    gt_left, gt_right = get_gt_left_right(gt_path)
    acc_left = compute_accuracy(patches_left_image, gt_left) 
    acc_right = compute_accuracy(patches_right_image, gt_right) 
    
    image_acc = (acc_left + acc_right) / 2 
    return image_acc
    


# In[12]:


# solve_image(image_path='images/image_10.jpg', gt_path='images/image_10.txt', use_hough=True, show_intermediate_results=True)


# In[237]:


images_paths = glob.glob('images/image_*.jpg')
accuracies = [solve_image(image_path=image_path, 
                      gt_path=image_path.replace('.jpg', '.txt'), 
                      use_hough=True,
                      show_intermediate_results=True)
                for image_path in images_paths]

print(np.mean(accuracies))


# In[ ]:




