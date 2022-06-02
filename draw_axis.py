from time import sleep
import cv2
from cv2 import aruco
import numpy as np
from camera_calibration import load_coefficients

ARUCO_SIZE = 5.5

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cameraMatrix, distCoeffs = load_coefficients(
        'calibration_chessboard.yml')
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        frame_markers = frame.copy()
        if ids is not None:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, ARUCO_SIZE, cameraMatrix, distCoeffs)

            #frame_markers = aruco.drawDetectedMarkers(
            #    frame_markers, corners, ids)

            for i in range(len(rvecs)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                frame_markers = cv2.drawFrameAxes(frame_markers, cameraMatrix, distCoeffs, rvec, tvec, ARUCO_SIZE/2)

        # Display the resulting frame
        cv2.imshow('frame', frame_markers)
        # cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord('q'):
            break

        # sleep(1/5)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
