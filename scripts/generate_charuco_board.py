import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.03, aruco_dict)
img = board.generateImage((1400, 1000))
cv2.imwrite("charuco_board_2.png", img)