import os
import glob
import pickle
import cv2 as cv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("All Modules Imported")

def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    total_correct = 0
    for i in range(1,51):
        filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")        
        filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")

        correct_flag = 1
        for row in range(1,10):
            p_line = p.readline()
            gt_line = gt.readline()
            # print(p_line)
            # print(gt_line)
            if (p_line[:10] != gt_line[:10]):
                print("Error in file {} at row: {}".format(filename_predictions, row))
                print("Prediction: ", p_line[:10])
                print("Ground Truth: ", gt_line[:10])
                correct_flag = 0

        p.close()
        gt.close()
        
        if verbose:
            print("Task 1 - Classic Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
        
        total_correct = total_correct + correct_flag
        points = total_correct * 0.05
        
    return total_correct, points

def evaluate_results_task2(predictions_path,ground_truth_path,verbose = 0):
    total_correct = 0
    for i in range(1,41):
        filename_predictions = predictions_path + "/" + str(i) + "_predicted.txt"
        p = open(filename_predictions,"rt")        
        filename_ground_truth = ground_truth_path + "/" + str(i) + "_gt.txt"
        gt = open(filename_ground_truth,"rt")
        correct_flag = 1
        for row in range(1,10):
            p_line = p.readline()
            gt_line = gt.readline()
            #print(p_line)
            #print(gt_line)
            if (p_line[:19] != gt_line[:19]):
                print("Error in file {} at row: {}".format(filename_predictions, row))
                # print(len(p_line[:19]))
                # print(len(gt_line[:19]))
                print("Prediction: ", p_line[:19])
                print("Ground Truth: ", gt_line[:19])
                correct_flag = 0        
        p.close()
        gt.close()
        
        if verbose:
            print("Task 2 - Jigsaw Sudoku: for test example number ", str(i), " the prediction is :", (1-correct_flag) * "in" + "correct", "\n")
        
        total_correct = total_correct + correct_flag
        points = total_correct * 0.05

        #break
        
    return total_correct, points

def preprocessing(image):
    image  = cv.medianBlur(image, 3)
    kernel = np.ones((7, 7), np.uint8)
    image  = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
    image  = cv.erode(image, kernel, iterations = 3)
    return image

def compute_mean_and_std(image_paths, display = False):
    largest = []
    for (step, path) in enumerate(image_paths):
        image = cv.imread(path)
        image = cv.resize(image, None, fx = 0.2, fy = 0.2)
        gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        preprocessed_image = preprocessing(gray)
        edges = cv.Canny(preprocessed_image, 150, 250)

        contours, hierarchy = cv.findContours(edges, 1, 2)

        max_area = 0
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
        plt.hist(largest, bins = 5)
        plt.show()

    return np.mean(largest), np.std(largest)


def generate_crop(contours, mean, std, n_std = 0.5):
    best_fit_contour, best_fit_idx = np.inf, 0
    threshold = (mean + (std * n_std))

    for (idx, cnt) in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)    

        computed_area = w * h
        if np.abs(threshold - computed_area) < best_fit_contour:
            best_fit_contour = np.abs(threshold - computed_area)
            best_fit_idx     = idx

    return best_fit_idx


def order_points(corner):
    corner  = corner.reshape(-1, 2)
    order   = np.zeros(4).astype(np.int64)

    summ = np.sum(corner, axis = 1)
    order[0] = np.argmin(summ)
    order[3] = np.argmax(summ)

    diff = np.diff(corner, axis=1)
    order[1] = np.argmin(diff)
    order[2] = np.argmax(diff)

    return corner[order]

def compute_colors_from_grid(grid):
    color = 0
    queue = []
    colors = np.zeros((9, 9), dtype = np.uint8)
    visited = np.zeros((9, 9), dtype = np.uint8)

    for k in range(9):
        for t in range(9):
            if visited[k][t] == 0:
                i = k; j = t
                color += 1
                queue.append((i, j))

                while len(queue) != 0:
                    (i, j) = queue.pop(0) 

                    visited[i][j] = 1
                    colors[i][j] = color

                    if grid[2 * i - 1][2 * j] != 1 and i - 1 >= 0 and visited[i - 1][j] == 0:
                        queue.append((i - 1, j))

                    if grid[2 * i][2 * j - 1] != 1 and j - 1 >= 0 and visited[i][j - 1] == 0:
                        queue.append((i, j - 1))

                    if grid[2 * i][2 * j + 1] != 1 and j + 1 < 9 and visited[i][j + 1] == 0:
                        queue.append((i, j + 1))

                    if grid[2 * i + 1][2 * j] != 1 and i + 1 < 9 and visited[i + 1][j] == 0:
                        queue.append((i + 1, j))

    return colors

def run_task_one(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL = False):
    image_paths = sorted(glob.glob(PATH_TO_IMAGES + "\\*.jpg"))

    mean, std = compute_mean_and_std(image_paths, display = False)

    #mean, std = 239115.43, 55094.31

    for (step, path) in enumerate(image_paths):
        print("Inference at file: {}".format(path))
        file_name = path.split(os.sep)[-1].split(".")[0]
        path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
        prediction_file = open(path_to_prediction_file, 'w')

        image = cv.imread(path)
        image = cv.resize(image, None, fx = 0.2, fy = 0.2)
        gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        preprocessed_image = preprocessing(gray)
        edges = cv.Canny(preprocessed_image, 150, 250)
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        idx = generate_crop(contours, mean, std)  
        cnt = contours[idx]

        perimeter = cv.arcLength(cnt, True)
        corners   = cv.approxPolyDP(cnt, 0.01 * perimeter, True)
        corners   = order_points(corners)

        generated_height, generated_width = 540, 540
        warp_start = corners.astype(np.float32)

        warp_end   = np.array([
            [0, 0],
            [generated_width, 0],
            [0, generated_height],
            [generated_width, generated_height]]).astype(np.float32)

        warp_matrix = cv.getPerspectiveTransform(warp_start, warp_end)
        warp = cv.warpPerspective(image, warp_matrix, dsize = (generated_width, generated_height)) 

        warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(warp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 20)

        threshold = 245
        step = generated_height // 9
        for row in range(0, thresh.shape[0], step):
            for column in range(0, thresh.shape[1], step):
                patch = thresh[row : row + step, column : column + step]
                patch = patch[10 : -10, 10 : -10]
                mean_color = patch.mean()
                
                if mean_color < threshold:
                    prediction_file.write("x")
                else:
                    prediction_file.write("o")

            if row == thresh.shape[0] - step and column == thresh.shape[1] - step:
                continue

            prediction_file.write('\n')

        prediction_file.close()

    if EVAL:
        total_correct, points = evaluate_results_task1(PATH_TO_PREDICTIONS, PATH_TO_IMAGES, verbose = 10)
        print("Total Corrects: {}, Task Points: {}".format(total_correct, points))
        exit(1)


def run_task_two(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL = False):
    image_paths = sorted(glob.glob(PATH_TO_IMAGES + "\\*.jpg"))

    mean, std = compute_mean_and_std(image_paths, display = False)

    #mean, std = 239115.43, 55094.31

    for (step, path) in enumerate(image_paths):
        # print("Inference at file: {}".format(path))
        # file_name = path.split(os.sep)[-1].split(".")[0]
        # path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
        # prediction_file = open(path_to_prediction_file, 'w')
        path = "train\\jigsaw\\28.jpg"
        image = cv.imread(path)
        image = cv.resize(image, None, fx = 0.2, fy = 0.2)
        gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        preprocessed_image = preprocessing(gray)
        edges = cv.Canny(preprocessed_image, 150, 250)
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        idx = generate_crop(contours, mean, std)  
        cnt = contours[idx]

        perimeter = cv.arcLength(cnt, True)
        corners   = cv.approxPolyDP(cnt, 0.01 * perimeter, True)
        corners   = order_points(corners)

        generated_height, generated_width = 540, 540
        warp_start = corners.astype(np.float32)

        warp_end   = np.array([
            [0, 0],
            [generated_width, 0],
            [0, generated_height],
            [generated_width, generated_height]]).astype(np.float32)

        warp_matrix = cv.getPerspectiveTransform(warp_start, warp_end)
        warp = cv.warpPerspective(image, warp_matrix, dsize = (generated_width, generated_height)) 

        # cv.imshow("warp", warp)
        # cv.waitKey(0)

        #break
        warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(warp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 37, 21)

        # cv.imshow("thresh", thresh)
        # cv.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)
        thresh  = cv.dilate(thresh, kernel, iterations = 1)

        kernel = np.ones((3, 3), np.uint8)
        thresh  = cv.erode(thresh, kernel, iterations = 1)

        thresh = cv.resize(thresh, (560, 560))
        thresh = thresh[10 : -10, 10 : -10]

         #break
        threshold = 235
        step = generated_height // 9

        margin = 15
        small_error = 10
        grid = np.zeros((18, 18), dtype = np.uint8)
        for row in range(0, thresh.shape[0], step):
            for column in range(0, thresh.shape[1], step):
                patch = thresh[row : row + step, column : column + step] 
                right = thresh[row + small_error : row + step - small_error, column + step - margin : column + step + margin]
                down  = thresh[row + step - margin : row + step + margin, column + small_error : column + step - small_error]

                i = row // step
                j = column // step

                if right.mean() < threshold:
                    grid[2 * i][2 * j + 1] = 1
                    grid[2 * i + 1][2 * j + 1] = 1
                
                if down.mean() < threshold:
                    grid[2 * i + 1][2 * j] = 1                    
                    grid[2 * i + 1][2 * j + 1] = 1

                # print("Right Mean: ", right.mean())
                # print("Down  Mean: ", down.mean())

                # cv.imshow("right", right)
                # cv.waitKey(0)

                # cv.imshow("down", down)
                # cv.waitKey(0)

        print(grid)

        colors = compute_colors_from_grid(grid)
        print(colors)

        cv.imshow("warp", warp)
        cv.waitKey(0)

        predictions = []
        thresh = cv.adaptiveThreshold(warp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 20)
        threshold = 250
        #print(threshold)
        step = generated_height // 9
        for row in range(0, thresh.shape[0], step):
            prediction = []
            for column in range(0, thresh.shape[1], step):
                patch = thresh[row : row + step, column : column + step]
                patch = patch[10 : -10, 10 : -10]
                mean_color = patch.mean()
                
                if mean_color < threshold:
                    prediction.append("x")
                else:
                    prediction.append("o")

            predictions.append(prediction)

        predictions = np.array(predictions)
        print(predictions)

        cv.imshow("morpho", thresh)
        cv.waitKey(0)

        # for i in range(9):
        #     for j in range(9):
        #         prediction_file.write(str(colors[i][j]))
        #         prediction_file.write(predictions[i][j])

        #     if i == 8 and j == 8:
        #         continue

        #     prediction_file.write('\n')

        # prediction_file.close()

        break

    if EVAL:
        total_correct, points = evaluate_results_task2(PATH_TO_PREDICTIONS, PATH_TO_IMAGES, verbose = 10)
        print("Total Corrects: {}, Task Points: {}".format(total_correct, points))
        exit(1)
  
def run_task_three(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL = False):
    image_paths = sorted(glob.glob(PATH_TO_IMAGES + "\\*.jpg"))

    mean, std = compute_mean_and_std(image_paths, display = False)

    #mean, std = 239115.43, 55094.31

    for (step, path) in enumerate(image_paths):
        print("Inference at file: {}".format(path))
        # file_name = path.split(os.sep)[-1].split(".")[0]
        # path_to_prediction_file = os.path.join(PATH_TO_PREDICTIONS, file_name + "_predicted.txt")
        # prediction_file = open(path_to_prediction_file, 'w')
        image = cv.imread(path)
        image = cv.resize(image, None, fx = 0.2, fy = 0.2)
        gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        preprocessed_image = preprocessing(gray)
        edges = cv.Canny(gray, 150, 250)

        cv.imshow("edges", edges)
        cv.waitKey(0)
        #break
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours()
    #     idx = generate_crop(contours, mean, std)  
    #     cnt = contours[idx]

    #     perimeter = cv.arcLength(cnt, True)
    #     corners   = cv.approxPolyDP(cnt, 0.01 * perimeter, True)
    #     corners   = order_points(corners)

    #     generated_height, generated_width = 540, 540
    #     warp_start = corners.astype(np.float32)

    #     warp_end   = np.array([
    #         [0, 0],
    #         [generated_width, 0],
    #         [0, generated_height],
    #         [generated_width, generated_height]]).astype(np.float32)

    #     warp_matrix = cv.getPerspectiveTransform(warp_start, warp_end)
    #     warp = cv.warpPerspective(image, warp_matrix, dsize = (generated_width, generated_height)) 

    #     # cv.imshow("warp", warp)
    #     # cv.waitKey(0)

    #     #break
    #     warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    #     thresh = cv.adaptiveThreshold(warp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 37, 21)

    #     # cv.imshow("thresh", thresh)
    #     # cv.waitKey(0)

    #     kernel = np.ones((5, 5), np.uint8)
    #     thresh  = cv.dilate(thresh, kernel, iterations = 1)

    #     kernel = np.ones((3, 3), np.uint8)
    #     thresh  = cv.erode(thresh, kernel, iterations = 1)

    #     thresh = cv.resize(thresh, (560, 560))
    #     thresh = thresh[10 : -10, 10 : -10]

    #      #break
    #     threshold = 235
    #     step = generated_height // 9

    #     margin = 15
    #     small_error = 10
    #     grid = np.zeros((18, 18), dtype = np.uint8)
    #     for row in range(0, thresh.shape[0], step):
    #         for column in range(0, thresh.shape[1], step):
    #             patch = thresh[row : row + step, column : column + step] 
    #             right = thresh[row + small_error : row + step - small_error, column + step - margin : column + step + margin]
    #             down  = thresh[row + step - margin : row + step + margin, column + small_error : column + step - small_error]

    #             i = row // step
    #             j = column // step

    #             if right.mean() < threshold:
    #                 grid[2 * i][2 * j + 1] = 1
    #                 grid[2 * i + 1][2 * j + 1] = 1
                
    #             if down.mean() < threshold:
    #                 grid[2 * i + 1][2 * j] = 1                    
    #                 grid[2 * i + 1][2 * j + 1] = 1

    #             # print("Right Mean: ", right.mean())
    #             # print("Down  Mean: ", down.mean())

    #             # cv.imshow("right", right)
    #             # cv.waitKey(0)

    #             # cv.imshow("down", down)
    #             # cv.waitKey(0)

    #     print(grid)

    #     colors = compute_colors_from_grid(grid)
    #     print(colors)

    #     cv.imshow("warp", warp)
    #     cv.waitKey(0)

    #     predictions = []
    #     thresh = cv.adaptiveThreshold(warp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 20)
    #     threshold = 250
    #     #print(threshold)
    #     step = generated_height // 9
    #     for row in range(0, thresh.shape[0], step):
    #         prediction = []
    #         for column in range(0, thresh.shape[1], step):
    #             patch = thresh[row : row + step, column : column + step]
    #             patch = patch[10 : -10, 10 : -10]
    #             mean_color = patch.mean()
                
    #             if mean_color < threshold:
    #                 prediction.append("x")
    #             else:
    #                 prediction.append("o")

    #         predictions.append(prediction)

    #     predictions = np.array(predictions)
    #     print(predictions)

    #     cv.imshow("morpho", thresh)
    #     cv.waitKey(0)

    #     # for i in range(9):
    #     #     for j in range(9):
    #     #         prediction_file.write(str(colors[i][j]))
    #     #         prediction_file.write(predictions[i][j])

    #     #     if i == 8 and j == 8:
    #     #         continue

    #     #     prediction_file.write('\n')

    #     # prediction_file.close()

    #     break

    # if EVAL:
    #     total_correct, points = evaluate_results_task2(PATH_TO_PREDICTIONS, PATH_TO_IMAGES, verbose = 10)
    #     print("Total Corrects: {}, Task Points: {}".format(total_correct, points))
    #     exit(1)      

if __name__ == "__main__":
    #EVAL = True
    #SCOPE = "classic"
    #PATH_TO_TRAIN  = "train\\"
    #PATH_TO_IMAGES = os.path.join(PATH_TO_TRAIN, SCOPE)
    #PATH_TO_PREDICTIONS = os.path.join("oof\\", SCOPE)
    #run_task_one(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL)

    # EVAL = False
    # SCOPE = "jigsaw"
    # PATH_TO_TRAIN  = "train\\"
    # PATH_TO_IMAGES = os.path.join(PATH_TO_TRAIN, SCOPE)
    # PATH_TO_PREDICTIONS = os.path.join("oof\\", SCOPE)
    # run_task_two(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL)

    EVAL = False
    SCOPE = "cube"
    PATH_TO_TRAIN  = "train\\"
    PATH_TO_IMAGES = os.path.join(PATH_TO_TRAIN, SCOPE)
    PATH_TO_PREDICTIONS = os.path.join("oof\\", SCOPE)
    run_task_three(PATH_TO_IMAGES, PATH_TO_PREDICTIONS, EVAL)




