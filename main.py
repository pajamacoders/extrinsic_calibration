import os
import cv2
import glob
import argparse
import logging
import numpy as np

logger = logging.Logger('calibration logger', level=logging.INFO)

pts=[] #[(tlx,tly), (rbx,rby), (tlx1, tly1),(brx1,bry1)]
drawing=False


def parse():
    parser = argparse.ArgumentParser(description = 'Calibration input parse')
    parser.add_argument('--input', type=str, default='./images/cam1', help='calibration input images directory')
    return parser.parse_args() 


def draw_roi(event, x,y, flags, param):
    global drawing, pts

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

def merge_close_point(points:list, dst_threshold:float=10.):
    """
    points: [(x,y), (x1,y1), ...]
    """
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

    return new_point_set

    

def nms(mat:np.ndarray, k: int):
    h,w = mat.shape
    pad_mat = np.zeros([np.ceil(i/k).astype(np.int)*k for i in mat.shape])
    pad_mat[:h,:w] = mat

    pad_h, pad_w = pad_mat.shape
    tiles = pad_mat.reshape(pad_h//k, k, pad_w//k, k).transpose(0,2,1,3).reshape(-1, k*k)
    rows = tiles.shape[0]
    mask = tiles==tiles.max(-1).reshape(rows, -1)
    tiles = tiles*mask

    #set all to 0 except first non zero element within each tile
    # nonzero_idx = mask.argmax(-1)
    # step = np.arange(nonzero_idx.shape[-1])*mask.shape[-1]+nonzero_idx
    # first_nonzero_mask = np.zeros(mask.reshape(-1).shape)
    # first_nonzero_mask[step]=1
    # first_nonzero_mask = first_nonzero_mask.reshape(tiles.shape)
    # tiles = first_nonzero_mask*tiles

    tiles = tiles.reshape(pad_h//k, pad_w//k, k,k).transpose(0,2,1,3).reshape(pad_h, pad_w)
    # 코너 근처의 다른 코너 삭제, 코너 정재를 할것이기 때문에 그냥 지워도 상관없음
   
    nms_mat = tiles[:h, :w]
    idxs = np.nonzero(nms_mat)
    return [p for p in zip(*idxs)]

def findCorners(img:np.ndarray, vis:bool = False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 1)
    blur = np.float32(blur)
    h,w = blur.shape
    dst = cv2.cornerHarris(blur,2,3,0.04)
    dst = dst*(dst>0.01*dst.max())
    points = nms(dst, 30)
    refined_points = merge_close_point(points)
    if vis:
        for y,x in refined_points:
            cv2.line(img, (max(x-3,0), y), (min(x+3,w-1),y), [0,0,255], 1)
            cv2.line(img, (x, max(y-3,0)),(x, min(y+3, h-1)), [0,0,255], 1)
        
        cv2.imshow('dst',img)
        cv2.waitKey(0)


if __name__=="__main__":
    
    args=parse()
    pathes = os.listdir(args.input)

    win_name = 'image'
    for path in pathes:
        img = cv2.imread(os.path.join(args.input, path))
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, draw_roi)
        tmp=img.copy()
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
        findCorners(roi, vis=True)
        # cv2.rectangle(img, (tlx, tly), (brx, bry), (0,0,255), 1)
        # cv2.imshow(win_name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

