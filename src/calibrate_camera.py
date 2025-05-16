import cv2
import numpy as np

chessboard_size = (7, 5)

images = [f'calib_imgs/calib_{str(i).zfill(2)}.jpg' for i in range(1, 25)]

objpoints = []
imgpoints = []

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(ret)
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    else:
        print(f'Chessboard not found in {fname}')

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)