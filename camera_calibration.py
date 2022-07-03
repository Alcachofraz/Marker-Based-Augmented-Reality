from time import sleep
from tkinter import N
import cv2
import numpy as np
import pathlib
from pynput import keyboard

"""
Press 'c' to take a photo of a chessboard and calibrate the camera.
Press 'q' to leave.
"""

SQUARE_SIZE = 2.7 # Size of chessboard squares in centimeters
WIDTH = 6 # Number of squares horizontally
HEIGHT = 9 # Number of squares vertically

def save_coefficients(mtx, dist, path):
    '''Save the camera matrix and the distortion coefficients to given [path].'''
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()


'''Calibrate camera using chessboard images obtained from camera.'''

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((HEIGHT * WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:WIDTH, 0:HEIGHT].T.reshape(-1, 2)

objp = objp * SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()  

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(1) == ord('c'):
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (WIDTH, HEIGHT), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # # Draw and display the corners
            # cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(10000)

            # Calibrate camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            # Save coefficients to a .yml file
            save_coefficients(mtx, dist, "calibration_chessboard.yml")
            print('Saved calibration_chessboard.yml')

            undistorted = cv2.undistort(gray, mtx, dist, None, None)
            cv2.imwrite('images/distorted.jpg', gray)
            cv2.imwrite('images/undistorted.jpg', undistorted)
            print('Saved distorted.jpg and undistorted.jpg')

    # Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
