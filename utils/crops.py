def generate_mean_crop(contours, mean, std, n_std = 0.5):
    best_fit_contour, best_fit_idx = np.inf, 0
    threshold = (mean + (std * n_std))

    for (idx, cnt) in enumerate(contours):
        computed_area = cv.contourArea(cnt)   
        perimeter = cv.arcLength(cnt, True)
        corners = cv.approxPolyDP(cnt, 0.01 * perimeter, True)

        if np.abs(threshold - computed_area) < best_fit_contour and len(corners) == 4:
            best_fit_contour = np.abs(threshold - computed_area)
            best_fit_idx     = idx
            
    print("generate_mean_crop -> ", best_fit_contour + threshold)
    return best_fit_idx


def generate_biggest_crop(contours):
    biggest_contour, biggest_idx = 0, 0

    for (idx, cnt) in enumerate(contours):
        computed_area = cv.contourArea(cnt)   
        perimeter = cv.arcLength(cnt, True)
        corners = cv.approxPolyDP(cnt, 0.01 * perimeter, True)

        if computed_area > biggest_contour and len(corners) == 4:
            biggest_contour = computed_area
            biggest_idx     = idx

    print("generate_biggest_crop -> ", biggest_contour)
    return biggest_idx

def generate_biggest_rect_crop(contours):
    biggest_contour, biggest_idx = 0, 0

    for (idx, cnt) in enumerate(contours):   
        x, y, w, h = cv.boundingRect(cnt)    
        computed_area = w * h

        if computed_area > biggest_contour:
            biggest_contour = computed_area
            biggest_idx     = idx

    print("generate_biggest_rect_crop -> ", biggest_contour)
    return biggest_idx

def generate_crop(contorurs, mean, std, n_std = 0.5):

    best_fit_contour, best_fit_idx = np.inf, 0
    threshold = (mean + (std * n_std))

    for (idx, cnt) in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)    

        computed_area = w * h
        if np.abs(threshold - computed_area) < best_fit_contour:
            best_fit_contour = np.abs(threshold - computed_area)
            best_fit_idx     = idx

    print("generate_crop -> ", best_fit_contour + threshold)
    return best_fit_idx

idx = generate_crop(contours, mean, std)
mean_idx = generate_mean_crop(contours, mean, std)
biggest_idx = generate_biggest_crop(contours)
biggest_rect_idx = generate_biggest_rect_crop(contours)

print(idx)
print(mean_idx)
print(biggest_idx) 
print(biggest_rect_idx)  