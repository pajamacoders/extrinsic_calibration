import os
from pydoc import visiblename
import cv2
import glob
import argparse
import logging
import numpy as np
import pickle
from intrinsic import cam1_intrinsic as cam1, cam2_intrinsic as cam2
from calibration import Calibration
from wrdpoints.cam1 import wrd_points
logger = logging.Logger('calibration logger', level=logging.INFO)

pts=[] #[(tlx,tly), (rbx,rby), (tlx1, tly1),(brx1,bry1)]
drawing=False


def parse():
    parser = argparse.ArgumentParser(description = 'Calibration input parse')
    parser.add_argument('--input', type=str, default='./images/cam1', help='calibration input images directory')
    return parser.parse_args() 


def draw_roi(event, x,y, flags, param):
    global drawing, pts
    tmp=param
    if event == cv2.EVENT_LBUTTONDOWN: #마우스를 누른 상태
        drawing = True   
        pts.append((x,y))

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스 이동
        if drawing==True: # 마우스를 누른 상태 일경우
            ix,iy = pts[-1]
            cv2.rectangle(tmp,(ix,iy),(x,y),(255,0,0),1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False;             # 마우스를 때면 상태 변경
        ix,iy = pts[-1]
        cv2.rectangle(tmp,(ix,iy),(x,y),(255,0,0),1)
        if abs((x-ix)*(y-iy)) > 500:
            pts.append((x,y))
        else:
            pts.pop()
            
    else:
        pass

def merge_close_point(corners:list, dst_threshold:float=10.):
    """
    corners: [(x,y), (x1,y1), ...]
    """
    
    points = corners.tolist() if isinstance(corners, np.ndarray) else corners
    new_point_set = []
    filtered_idx = []
    for i in range(len(points)):
        idxs=[]
        if i not in filtered_idx:
            for j in range(i+1, len(points)):
                dst = np.sqrt((points[i][0]-points[j][0])**2+(points[i][1]-points[j][1])**2)
                if dst<=dst_threshold:
                    idxs.append(j)
        if idxs:
           filtered_idx+=idxs
        else:
            new_point_set.append(points[i])

    return np.array(new_point_set).reshape(-1,2) if isinstance(corners, np.ndarray) else new_point_set

    

def nms(mat:np.ndarray, k: int):
    h,w = mat.shape
    pad_mat = np.zeros([np.ceil(i/k).astype(np.int)*k for i in mat.shape])
    pad_mat[:h,:w] = mat

    pad_h, pad_w = pad_mat.shape
    tiles = pad_mat.reshape(pad_h//k, k, pad_w//k, k).transpose(0,2,1,3).reshape(-1, k*k)
    rows = tiles.shape[0]
    mask = tiles==tiles.max(-1).reshape(rows, -1)
    tiles = tiles*mask
    tiles = tiles.reshape(pad_h//k, pad_w//k, k,k).transpose(0,2,1,3).reshape(pad_h, pad_w)
    # 코너 근처의 다른 코너 삭제, 코너 정재를 할것이기 때문에 그냥 지워도 상관없음
   
    nms_mat = tiles[:h, :w]
    idxs = np.nonzero(nms_mat)
    return [p for p in zip(*idxs)]

def remove_out_lier(corners, canny):
    h,w = canny.shape
    new_points = []
    for x,y in corners:
        ty = int(max(y-2, 0))
        by = int(min(y+3, h-1))
        tx = int(max(x-2, 0))
        bx = int(min(x+3, w-1))
        edges = canny[ty:by,tx:bx].sum()
        if edges > 2:
            new_points.append([x,y])
    return np.array(new_points).reshape(-1,2)


def get_y_coordinates(param, x):
    y=None
    if param[1]!=0.:
        y=(-x*param[0]-param[2])/param[1]
    return y

def find_lines(blur, canny, vis=True):
    dst = cv2.Sobel(blur, cv2.CV_32FC1, 0,1)
    dst = np.absolute(dst)
    dst = dst*(dst>50)
    dst= np.clip(dst*canny,0,255).astype(np.uint8)
    h,w = canny.shape

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = int(w*0.4)  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = int(w*0.3)  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
     # creating a blank to draw lines on
    lines = cv2.HoughLinesP(dst, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
    line_models = []
    if lines.shape[0]>0 :

        lines = sorted(lines.squeeze(), key= lambda x: (x[1]+x[3]/2.))
        print('lines:', lines)

        for x1, y1, x2, y2 in lines:
                param = np.cross([x1,y1,1], [x2,y2,1])
                line_models.append((param, (x1, y1), (x2, y2)))
        if vis:
            line_image = canny.copy()
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
            for param, sp,ep in line_models:
                    py1 = get_y_coordinates(param, sp[0])
                    py2 = get_y_coordinates(param, ep[0])
                    cv2.line(line_image, (sp[0], int(py1)), (ep[0], int(py2)), (0, 0, 255), 1)
            cv2.imshow('dst', dst)
            cv2.imshow('line', line_image)
            cv2.waitKey(0)
    return line_models

def clustering_corners(blur, canny, corners, size:tuple[int,int]):
    tmp_pts=np.append(corners, np.ones(corners.shape[0]).reshape(-1,1), axis=-1)
    lines = find_lines(blur, canny)
    sorted_lines = []
    if len(lines)==size[1]:
        for line, sp, ep in lines:
            dist = line.dot(tmp_pts.transpose())
            dist = dist**2/(line[:2]**2).sum()
            mask = dist<=4.
            cluster = corners[mask]
            cluster = sorted(cluster, key=lambda x: x[0])
            if len(cluster) != size[1]:
                raise Exception('Fail to find proper number of points that belong to pattern.')
            sorted_lines.append(cluster)
    else:
        raise Exception('fail clustering. Lines are not found.')
    return np.array(sorted_lines).reshape(-1,2)
        
def findCorners_v2(img:np.ndarray, size:tuple[int,int], vis:bool=False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 1)
    canny = cv2.Canny(blur, 30,200)
    blur = np.float32(blur)
    h,w = blur.shape
    dst = cv2.cornerHarris(blur,5,5,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    refined_points = merge_close_point(centroids)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(blur,np.float32(refined_points),(5,5),(-1,-1),criteria)
    corners = remove_out_lier(corners, canny)
    try:
        corners = clustering_corners(blur, canny, corners, size)
    except Exception as e:
        print(e)
        exit(0)
    if vis:
        for x,y in centroids.astype(np.int32):
            cv2.line(img, (max(x-3,0), y), (min(x+3,w-1),y), [0,0,255], 1)
            cv2.line(img, (x, max(y-3,0)),(x, min(y+3, h-1)), [0,0,255], 1)
        
        for x,y in corners.astype(np.int32):
            cv2.line(img, (max(x-3,0), y), (min(x+3,w-1),y), [0,255,0], 1)
            cv2.line(img, (x, max(y-3,0)),(x, min(y+3, h-1)), [0,255,0], 1)
        cv2.imshow('dst',img)
        cv2.waitKey(0)
    
    return corners

def extract_points(img_path, cls_fname, far_fname):

    win_name = 'image'
    for path, fname in zip(img_path, [cls_fname, far_fname]):
        img = cv2.imread(path)
        cv2.namedWindow(win_name)
        tmp=img.copy()
        cv2.setMouseCallback(win_name, draw_roi, tmp)

        while 1:
            cv2.imshow(win_name, tmp)
            k = cv2.waitKey(20) & 0xFF
            if k==27:
                break
            elif k==99:
                print('b:',pts)
                pts.pop()
                print('a:',pts)
            
        print(pts)
        brx, bry = pts.pop()
        tlx, tly = pts.pop()
        roi = img[tly:bry+1, tlx:brx+1]
        corners = findCorners_v2(roi,(3,3), vis=False)
        corners+=np.array([tlx,tly]).reshape(1,2)
        h,w,c = img.shape
        for x,y in corners.astype(np.int32):
            cv2.line(img, (max(x-3,0), y), (min(x+3,w-1),y), [0,255,0], 1)
            cv2.line(img, (x, max(y-3,0)),(x, min(y+3, h-1)), [0,255,0], 1)

        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        with open(fname, 'wb') as f:
            pickle.dump(corners, f)

if __name__=="__main__":
    
    args=parse()
    pathes = os.listdir(args.input)
    pathes=sorted(pathes)
    pathes = [os.path.join(args.input, path) for path in pathes]
    cls_fname, far_fname = 'close.ptk', 'far.ptk'
    calibrator = Calibration(args.input)
    while 1:
        print('select menu:')
        print('1. extract corner points')
        print('2. compute extrinsic parameter')
        print('3. compute intrinsic param')
        print('4. undistort image')
        print('5. quit')
        ch = input('select: ')
        if ch.isdigit():
            ch=int(ch)
        else:
            print('numeric value is required.')

        if ch==1:
            extract_points(pathes, cls_fname, far_fname)
        elif ch==2:
            win_name = 'image'
            cls = cv2.imread(pathes[0])
            far = cv2.imread(pathes[1])
            cam = cam1
            k = cam.cam_K
            distCoef = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            with open(cls_fname, 'rb') as f:
                cls_points = pickle.load(f)
            with open(far_fname, 'rb') as f:
                far_points = pickle.load(f)
            calibrator.compute_extrinsic(cls_points, far_points, k, distCoef,  wrd_points)
            calibrator.get_translation()
            calibrator.output_result('cam1.calib')
            calibrator.display(k, distCoef, wrd_points, len(cls_points), cls, far)
        elif ch == 3:
            
            calibrator.compute_intrinsic()
        elif ch==4:
            k =cam1.cam1_K
            distCoef = np.array([cam1.k1, cam1.k2, cam1.p1, cam1.p2], dtype=np.float32)
            img = cv2.imread('test.png')
            dst = cv2.undistort(img, k, distCoef, None)
            cv2.imshow('src', img)
            cv2.imshow('dst', dst)
            cv2.waitKey(0)
        elif ch ==5:
            exit(0)
        
    

