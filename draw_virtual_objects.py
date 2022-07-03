from time import sleep
import cv2
from cv2 import aruco
import numpy as np

"""
Press 'c' to take a picture and save it to images/virtual_objects.jpg.
"""

ARUCO_SIZE = 5.5

virtual_objects = [
    cv2.imread('./images/album_covers/a_dramatic_turn_of_events.jpg'),
    cv2.imread('./images/album_covers/distant_satellites.jpg'),
    cv2.imread('./images/album_covers/in_absentia.jpg'),
    cv2.imread('./images/album_covers/octavarium.jpg'),
    cv2.imread('./images/album_covers/panther.jpg'),
    cv2.imread('./images/album_covers/remedy_lane.jpg'),
    cv2.imread('./images/album_covers/the_perfect_element.jpg'),
    cv2.imread('./images/album_covers/the_raven_that_refused_to_sing_and_other_stories.jpg'),
    cv2.imread('./images/album_covers/the_optimist.jpg'),
    cv2.imread('./images/album_covers/the_similitude_of_a_dream.jpg'),
    cv2.imread('./images/album_covers/weather_systems.jpg'),
    cv2.imread('./images/album_covers/whirlwind.jpg'),
]


def load_coefficients(path):
    '''Load camera matrix and distortion coefficients from given [path].'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

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
        if ids is not None and len(corners) != 0:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, ARUCO_SIZE, cameraMatrix, distCoeffs)
                
            for i in range(len(corners)):
                bbox = corners[i]
                tl = bbox[0][0][0], bbox[0][0][1]
                tr = bbox[0][1][0], bbox[0][1][1]
                br = bbox[0][2][0], bbox[0][2][1]
                bl = bbox[0][3][0], bbox[0][3][1]
                object = virtual_objects[ids[i][0] - 1]
                h, w, c = object.shape
                pts1 = np.array([tl, tr, br, bl])
                pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
                matrix, _ = cv2.findHomography(pts2, pts1)
                imgOut = cv2.warpPerspective(object, matrix, (frame_markers.shape[1], frame_markers.shape[0]))

                # Keep real background:
                cv2.fillConvexPoly(frame_markers, pts1.astype(int), (0, 0, 0))
                frame_markers = frame_markers + imgOut

        if cv2.waitKey(1) == ord('c'):     
            cv2.imwrite('images/virtual_objects.jpg', frame_markers)
            print('Saved axis.jpg')

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
