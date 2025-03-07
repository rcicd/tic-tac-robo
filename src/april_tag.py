import cv2
import cv2.aruco as aruco
import numpy as np


fx = 1.28945513e+03
cx = 1.04571605e+03
fy = 1.27196805e+03
cy = 5.92741724e+02
cameraMatrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

k1 = -0.07995083
k2 = -1.18597715
p1 = -0.01550925
p2 = -0.00882771
k3 = 1.78955232
distCoeffs = np.array([k1, k2, p1, p2, k3]) 

marker_size = 0.065


frame = cv2.imread('apriltag_samples/3.jpg')


def get_all_markers(frame = frame, 
                    cameraMatrix = cameraMatrix, 
                    distCoeffs = distCoeffs, 
                    marker_size = marker_size, dev = True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    parameters = aruco.DetectorParameters()
    result_list = [] 
    

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if dev:
        print("Found markers (ids):", ids)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 
                                                          marker_size, 
                                                          cameraMatrix, 
                                                          distCoeffs)
        for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            if dev:
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
        
            R, _ = cv2.Rodrigues(rvec)
            
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6
            
            if not singular:
                x_angle = np.arctan2(R[2, 1], R[2, 2])
                y_angle = np.arctan2(-R[2, 0], sy)
                z_angle = np.arctan2(R[1, 0], R[0, 0])
            else:
                x_angle = np.arctan2(-R[1, 2], R[1, 1])
                y_angle = np.arctan2(-R[2, 0], sy)
                z_angle = 0
                
            distance = np.linalg.norm(tvec)
            
            if dev:
                print("Marker ID:", ids[idx])
                print("Orientation (radians): X: {:.2f}, Y: {:.2f}, Z: {:.2f}"
                      .format(x_angle, y_angle, z_angle))
                print("Distance to marker: {:.2f} m".format(distance))
                print("-" * 40)
            
            data = {
                "orientation": (x_angle, y_angle, z_angle),
                "distance": distance
            }
            result_list.append((ids[idx], data))
    
    if dev:
        cv2.imshow("Detected Markers", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_list

def get_relative_pose(marker1, marker2):
    rvec1, tvec1 = marker1
    rvec2, tvec2 = marker2

    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    R_rel = np.dot(R1.T, R2)
    
    tvec_rel = np.dot(R1.T, (tvec2 - tvec1))
    
    rvec_rel, _ = cv2.Rodrigues(R_rel)

    return rvec_rel, tvec_rel

if __name__ == "__main__":
    q = get_all_markers(dev=False)
    m159 = q[0]
    m161 = q[1]
    m159_id, m159_data = m159
    m161_id, m161_data = m161
    rvec_rel, tvec_rel = get_relative_pose((m159_data["orientation"], m159_data["distance"]),
                                           (m161_data["orientation"], m161_data["distance"]))
    print("Relative position of markers with id 159 and 161:")
    print("Orientation by axis (radians):", rvec_rel)
    print("Distance between markers: {:.2f} m".format(np.linalg.norm(tvec_rel)))