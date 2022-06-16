import os
import cv2
import numpy as np 
import json
class Calibration:
    def __init__(self, root:str, size:int =50):
        self.pathes = [os.path.join(root, path) for path in os.listdir(root)]
        self.objpoints = np.array([[i*50,j*50, 0] for i in range(5) for j in range(8)], dtype = np.float32).reshape(-1,3)


    def compute_intrinsic(self):
        #read images
        imgs = [cv2.imread(path) for path in self.pathes]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0) # ( type, max_iter, epsilon )
        img_pts = []
        for img in imgs:
            ret, corners = cv2.findChessboardCorners(img, patternSize=(8,5))
            if ret:
                rf_corners = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners, (10,10), (-1,-1), criteria)
                img_pts.append(rf_corners)
                cv2.drawChessboardCorners(img, (8,5), rf_corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(1)
        
        obj_pts = [self.objpoints]*len(img_pts)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (640,480), None, None)
        print(ret)
        
    def compute_extrinsic(self, cls_points, far_points, k, distCoef, wrd_points):
   
        img_points  = np.append(cls_points, far_points, axis=0)#np.array(cls_points+far_points, dtype=np.float32).reshape(-1,2)
        #img_points  = np.array(cls_points, dtype=np.float32).reshape(-1,2)
        #ret, self.rvec, self.tvec = cv2.solvePnP(wrd_points, img_points, k, distCoef, flags=cv2.SOLVEPNP_EPNP)
        ret, self.rvec, self.tvec = cv2.solvePnP(wrd_points, img_points, k, distCoef, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # def decomposeResult(self):
    #     self.Rw2c = np.linalg.inv(cv2.Rodrigues(self.rvec)[0])
    #     yaw = np.rad2deg(np.arctan(self.Rw2c[1,0]/self.Rw2c[0,0]))
    #     pitch = np.rad2deg(np.arctan(self.Rw2c[2,1]/self.Rw2c[2,2]))
    #     roll = np.rad2deg(np.arcsin(-self.Rw2c[2,0]))
    #     print(f'roll:{roll:.3f}, pitch:{pitch:.3f}, yaw:{yaw:.3f}')

    
    def get_translation(self):
        self.Rw2c = np.linalg.inv(cv2.Rodrigues(self.rvec)[0])
        self.z,self.y,self.x = self.decompose_Rotmat2zyx(self.Rw2c)
        self.trans=-np.dot(self.Rw2c, self.tvec)
        print(f'x:{self.trans[0,0]:.2f}mm, y:{self.trans[1,0]:.2f}mm, z:{self.trans[2,0]:.2f}m')

    def output_result(self, fname):
        output = {'rvec': self.rvec.tolist(),
        'tvec': self.tvec.tolist(),
        'lateral': self.trans[0,0],
        'longitudinal': self.trans[1,0],
        'height': self.trans[2,0],
        'z':self.z,
        'y':self.y,
        'x':self.x}
        with open(fname,'w') as f:
            json.dump(output, f)

    def display(self, k, distCoef, wrd_points, length, cls_img=None, far_img=None):

        if cls_img is not None:
            repo_cls_img, jacobian = cv2.projectPoints(wrd_points[:length], self.rvec, self.tvec, k, distCoef)
            for p in repo_cls_img.squeeze():
                cv2.line(cls_img, (int(p[0])-2, int(p[1])), (int(p[0])+2, int(p[1])), (0,0,255), 1)
                cv2.line(cls_img, (int(p[0]), int(p[1])-2), (int(p[0]), int(p[1])+2), (0,0,255), 1)
            cv2.imshow('cls', cls_img)

        if far_img is not None:
            repo_far_img, jacobian = cv2.projectPoints(wrd_points[length:], self.rvec, self.tvec, k, distCoef)
            for p in repo_far_img.squeeze():
                cv2.line(far_img, (int(p[0])-2, int(p[1])), (int(p[0])+2, int(p[1])), (0,0,255), 1)
                cv2.line(far_img, (int(p[0]), int(p[1])-2), (int(p[0]), int(p[1])+2), (0,0,255), 1)
            cv2.imshow('far', far_img)

        if cls_img is not None or far_img is not None:
            cv2.waitKey(0)

    def getRotMatFromZYX(self, z,y,x):
        Rz, Ry, Rx = np.eye(3), np.eye(3), np.eye(3)
        th_z = np.deg2rad(z)
        th_y = np.deg2rad(y)
        th_x = np.deg2rad(x)
        Rz[0,0] = np.cos(th_z)
        Rz[0,1] = np.sin(th_z)
        Rz[1,0] = -np.sin(th_z)
        Rz[1,1] = np.cos(th_z)

        Ry[0,0] = np.cos(th_y)
        Ry[0,2] = -np.sin(th_y)
        Ry[2,0] = np.sin(th_y)
        Ry[2,2] = np.cos(th_y)

        Rx[1,1] = np.cos(th_x)
        Rx[1,2] = np.sin(th_x)
        Rx[2,1] = -np.sin(th_x)
        Rx[2,2] = np.cos(th_x)
        R = np.matmul(Rz.T,np.matmul(Ry.T,Rx.T))
        return R

    def decompose_Rotmat2zyx(self, rot):
        z = np.rad2deg(np.arctan2(rot[1,0],rot[0,0]))
        y = np.rad2deg(np.arcsin(rot[2,0]))
        x = np.rad2deg(np.arctan2(rot[2,1],rot[2,2]))
        print(f'z:{z:.3f}, y:{y:.3f}, x:{x:.3f}')
        return z,y,x
            