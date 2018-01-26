import numpy as np
import h5py
import cv2
import tifffile as tif
import math
import os
import sys

# Constants
bounding_region = 30    # ROI
thresh_val = 150   # 
lower_area_limit = 100 # contur 정하는 길이??
upper_area_limit = 25000
circ_score_thresh = 1.55

def import_eye(filename):
    '''
    Extracts data from .mat.
    '''
    if '.mat' in filename:
        filename = os.path.splitext(filename)[0]
    data = h5py.File(filename + '.mat')
    ## h5py: mat 데이터를 numpy array 데이터로 변환
    
    eye_data = np.squeeze(np.array(data['data'])).transpose(0,2,1)
    ## numpy.squeeze: Remove single-dimensional entries from the shape of an array.
    ## numpy.transpose(a, axes=None) : Permute the dimensions of an array.
    ## -> 데이터의 순서 배치를 바꾼다. 뒤에서 depth, height, width 순서로 받기 위함인듯

    return eye_data
## eye_data는 tiff를 이루고있는 모든 frame에 대한 data 임

def find_pupil_contours(eye_data, equalize=True):
    '''
    Outputs array containing pupil contour for each frame.
    '''
    # Get shape of data
    depth, height, width = eye_data.shape
    # Calculate coordinates for center of frame
    center = np.array(eye_data.shape[1:])/2
    # Calculate coordinates for center of cropped frame
    cropped_center = np.array([bounding_region, bounding_region])/2
    ## 전역변수로 정의된 bounding_region을 통해 ROI, 즉 관심을 가져야할 영역만을 지정해준다.
    
    # Generate slices to crop frame
    height_slice = slice(center[0]-bounding_region, center[0]+bounding_region)
    width_slice = slice(center[1]-bounding_region, center[1]+bounding_region)
    # Calculate X distance between full frame edge and cropped edge
    x_offset = (width - 2*bounding_region)/2
    # Calculate Y distance between full frame edge and cropped edge
    y_offset = (height - 2*bounding_region)/2
    
    # Create adaptive histogram equalization params
    ## clahe라는 함수? 로 히스토그램 equalization 을 할 수 있게 한다.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    ## 여기까지는 그냥 윤곽선을 잘 도출하기 위한 관심영역 지정을 한 것이다.
    
    # 이제 곧 나올 윤곽선을 위해 list 선언
    contour_list = []
    

    for frame in eye_data:
        if equalize:
            # Apply adaptive histogram equalization
            frame = clahe.apply(frame)
        # Crop frame to restrict pupil search region
        cropped_frame = frame[height_slice, width_slice]
        # Apply threshold to frame
        ret, thresh = cv2.threshold(cropped_frame.copy(), thresh_val, 255, cv2.THRESH_BINARY)
        
        ## 여기까지는 윤곽선을 잘 찾기위해 영상의 색을 보정하고, binary 이미지로 바꿔준 것.
        
        
        # Find contours
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find convex hull of each contour
        ## 도출된 윤곽선 점 list를 따라 도형?? 을 그려주는 것
        hulls = [cv2.convexHull(contour) for contour in contours]
        # Find area of each hull
        areas = [cv2.contourArea(hull) for hull in hulls]
        # Filter hulls by area size
        ## 그린 도형(hull)의 크기가 적절하면 남긴다.
        hulls = [hull for hull,area in zip(hulls,areas) if area > 100 and area < 25000]
        # Find moments of remaining hulls
        moments = [cv2.moments(hull) for hull in hulls]
        # Calculate centroid of remaining hulls
        ## 남겨진 적절한 도형(이게 이제 동공이 되겠지??) 안에서 중심점 후보 list를 구한다.
        centroids = [[int(m['m10']/m['m00']),int(m['m01']/m['m00'])] if m['m00'] != 0 else [0,0] for m in moments]

        
        ## 중심점 후보 list에서 중심점을 구하는 부분!
        if len(centroids):
            # Select contour with centroid closest to center of frame
            ## 위에서 구한 cropped_center와 가장 가까운 점을 구하는 듯??
            dist_list = np.sum(np.abs(centroids - cropped_center), 1)
            center_contour_index = np.argmin(dist_list)
            center_contour = hulls[center_contour_index]
            center_centroid = centroids[center_contour_index]
            
            # Correct contour offset
            center_contour[:,0][:,0] = center_contour[:,0][:,0] + x_offset
            center_contour[:,0][:,1] = center_contour[:,0][:,1] + y_offset
            
            ## contours_list라는 윤곽선 점들의 모임 안에 그 윤곽선 안의 중심점을 추가해준거임!
            contour_list.append(center_contour)
        else:
            # No contours found, fill with None
            contour_list.append(None)

    return contour_list

def annotate_eye(eye_data, contours, centroid=True, score=True):
    '''
    Draws contours onto each frame and returns data in RGB format.
    '''
    ## 추출된 contour list를 원본 영상 위에 putText를 이용하여 써준다.
    ## return : 원본데이터와 contour 가 합쳐져서 나옴
    
    rgb_eye_data = []
    depth, height, width = eye_data.shape
    for frame, contour in zip(eye_data, contours):
        # Convert to rgb
        rgb_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
        if not isinstance(contour, np.ndarray):
            rgb_eye_data.append(rgb_frame)
            continue
        # Calculate contour centroid
        M = cv2.moments(contour)
        if M['m00']:
            centroid = [int(M['m10']/M['m00']), int(M['m01']/M['m00'])]
        else:
            centroid = [0, 0]
        # Calculate circularity score
        center_surround_distances = [
                math.hypot(
                    point[0][0] - centroid[0], point[0][1] - centroid[1]
                    ) for point in contour
                ]
        min_dist = min(center_surround_distances)
        max_dist = max(center_surround_distances)
        circ_score = max_dist/min_dist
        color = (0, 255, 0) if circ_score < circ_score_thresh else (255, 0, 0)

        # Draw contour
        img = cv2.drawContours(rgb_frame, [contour], 0, color, 2)

        if centroid:
            # Draw contour centroid
            cv2.rectangle(rgb_frame,
                (centroid[0] - 2, centroid[1] - 2),
                (centroid[0] + 2, centroid[1] + 2),
                color,
                -1)

        if score:
            # Draw circularity score
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_frame, str(circ_score), (10, int(.9*height)), font, 1, color, 2)

        rgb_eye_data.append(rgb_frame)
    return np.array(rgb_eye_data)

def main():
    filename = sys.argv[1]
    basename = os.path.splitext(filename)[0]
    print('Opening eye data...')
    eye_data = import_eye(filename)
    print('Extracting pupil contours...')
    contours = find_pupil_contours(eye_data)
    np.save('{}_contours'.format(basename), contours)
    
    if '-write' in sys.argv:
        print('Annotating data...')
        annotated_data = annotate_eye(eye_data, contours)
        print('Saving annotated data...')
        tif.imsave('{}_annotated.tif'.format(basename), annotated_data)
    print('Finished')

if __name__ == '__main__':
    main()
