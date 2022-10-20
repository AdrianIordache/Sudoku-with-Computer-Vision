# Sudoku with Computer Vision
***Exploring non-learning (mostly) computer vision techniques for various tasks applied to the Sudoku game***

## Task 1: Extracting configurations of Classic Sudoku puzzles
In the first task you are asked to write a program that processes an input image containing a Classic Sudoku puzzle and outputs the configuration of the puzzle by determining whether or not a cell contains a digit.

We mark empty cells with letter ’o’ and the filled in cells with letter ’x’. The training data consists of 50 training examples. Each training example (an image obtained by taking a photo with the mobile phone) contains one Classic Sudoku puzzle, centered, usually axis aligned or with small rotations with respect to the Ox and Oy axis.

![](https://github.com/AdrianIordache/Sudoku-with-Computer-Vision/blob/main/images/task-1.png)

## Proposed Solution

For each image that we have:
  - Choose a robust preprocessing way to be able to detect the sudoku square without noise

  - Apply a Canny Filter for edge detection (it's easier to find contours there)
  
  - Generate all posible contours on that image
  
  - Pick the "best" contour (this ideally represents the Sudoku Square, attention, this might not be the largest square in the image)
    
  - Apply a perspective transform based on the (ordered) corners of that contour to remove possible rotations
 
  - Apply some more preprocessing to the obtained image for the inference step
  
  - Based on the fact that we now the obtained image should be a 9x9 grid, iterate through patches of pixels
  
  - Based on a chosen threshold and the mean intensity of the patch, predict if the image contains or not some digit

## Results: 50/50 Correct Test Examples

## Task 2: Extracting configurations of Jigsaw Sudoku puzzles

In the second task you are asked to write a program that processes an input image containing a Jigsaw Sudoku puzzle and outputs the configuration of the puzzle by: (1) determining the irregular shape regions in the puzzle; (2) determining whether or not a cell contains a digit. For this task, we mark all cells with a string of length two: the digit (1 to 9) corresponding to the irregular shape region where the cell is positioned and a letter (’o’ or ’x’) specifying whether or not the cell is empty.

The irregular shape regions from the puzzle are separated by bold borders and sometimes (in the colored puzzles) contain cells with the same color. For determining the digit corresponding to a cell in an irregular shape regions in a Jigsaw puzzle we use the following simple algorithm:

(i) we process the cells from left to right and top to bottom;

(ii) the top left cell gets digit 1 as it is part of region 1;

(iii) we assign the same digit for all cells in the same region;

(iv) the first cell in the next region gets the increased digit (we move to the next region).

The training data consists of 40 training examples (20 colored and 20 black and white Jigsaw Sudoku puzzles). Each training example (an image obtained by taking a photo with the mobile phone) contains one Jigsaw Sudoku puzzle, either colored or black and white, centered, usually axis aligned or with small rotations with respect to the Ox and Oy axis. A colored Jigsaw Sudoku puzzle will always contain regions of three possible colors: blue, yellow and red.

![](https://github.com/AdrianIordache/Sudoku-with-Computer-Vision/blob/main/images/task-2.png)

## Proposed Solution

Based on the solution from Task 1, now we have a robust method to detect the Sudoku Square.

### For the second task we need to find a way remove thin edges without losing thick edges, which describe each region.

Based on the new preprocessed image (only with thick edges) and the fact that we know that our image contains a 9x9 grid with fixed positions for possible borders, we can predict based on the intensity of a patch of pixels and some threshold if a certain position contains a region border.

This way we can generate a grid matrix with borders that will be used for predicting each region.

For infering digits will use the same approach as in Task 1.

### The algorithm becomes

For each image that we have:
  - Choose a robust preprocessing way to be able to detect the sudoku square without noise
  - Apply a Canny Filter for edge detection (it's easier to find contours there)
  - Generate all posible contours on that image
  - Pick the "best" contour (this ideally represents the Sudoku Square, attention, this might not be the largest square in the image)
  - Apply a perspective transform based on the (ordered) corners of that contour to remove possible rotations
  - Apply some more preprocessing to the obtained image for removig thin edges, but keeping region borders
  - Based on the fact that we now the obtained image should be a 9x9 grid, iterate through patches of pixels
  - Based on a chosen threshold and the mean intensity of the patch, predict if the image contains or not a region border
  - Generate a grid matrix with borders for generating regions
  - Apply a coloring algorithm for predict regions based on the grid
  - Use the methodology from task 1 to predict possible digits in the Sudoku Square

## Results: 40/40 Correct Test Examples

## Task 3: Assembling a Sudoku Cube

In the third task you are asked to write a program that processes an input image containing three sides (each side is a sudoku puzzles) of a Sudoku Cube and outputs the coresponding Sudoku Cube by:

(1) localizing the three sudoku puzzles in the image that form the sides of the Sudoku Cube;

(2) inferring their position in the Sudoku Cube using the constraint that the digits on the common edge of two sides must be the same number;

(3) warping each side on the corresponding side of a given template for the Sudoku Cube.

The training data consists of 10 training examples. Each training example (an image 1500 × 1500 generated on the computer) contains three sides of a Sudoku Cube . They are scattered around the image and usually rotated with respect to the axis. First, you have to find the three sudoku puzzles in the image, recognize the digits in each puzzle and solve the simple problem of matching the sides in the Sudoku cube. Please note that there is one solution.

Then you have to warp the puzzles on the template in order to obtain the desired result.

For warping, you are allowed to manually annotate points on the template for warping (but, of course, your are not allowed to to the same thing on the test images as we want your method to be automatically) such that it is easy to map each puzzle found in the image on the corresponding side of the Cube.

![](https://github.com/AdrianIordache/Sudoku-with-Computer-Vision/blob/main/images/task-3.1.png)
![](https://github.com/AdrianIordache/Sudoku-with-Computer-Vision/blob/main/images/task-3.2.png)


## Proposed Solution

For this task we will change the methodology for initial preprocessing, using just some 3x3 dilation for more pronounced edges, and choosing the biggest three contours based on the bounding rectangle area.

After the standard perspective transform, we will use a [Convolutional Neural Network trained some time ago](https://github.com/AdrianIordache/DeepLearning-In-Pytorch/blob/master/Transfer-Learning-On-Counting-MNIST-Dataset/Assignment-2.ipynb). 

The accuracy at that time was about 99.3%, to improve that I decided to use 3 crops of TTA (Test Time Augmentation) averaging them before the argmax layer to obtain the final prediction.

For each sudoku square we will compare the top row with with each sudoku square buttom row to obtain a reindexing of squares for the final prediction file.

Based on the same indexing will wrap the corners of detected squares to the annotated corners of the template image.

## Results: 10/10 Correct Test Examples
