import numpy as np
import cv2
import operator
from Sunfish import *


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Below are the Sunfish data tables - %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e6

# This constant controls how much time we spend on looking for optimal moves.
NODES_SEARCHED = 1e4

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
MATE_VALUE = 30000

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  # 0 -  9
    '         \n'  # 10 - 19
    ' rnbqkbnr\n'  # 20 - 29
    ' pppppppp\n'  # 30 - 39
    ' ........\n'  # 40 - 49
    ' ........\n'  # 50 - 59
    ' ........\n'  # 60 - 69
    ' ........\n'  # 70 - 79
    ' PPPPPPPP\n'  # 80 - 89
    ' RNBQKBNR\n'  # 90 - 99
    '         \n'  # 100 -109
    '          '  # 110 -119
)

###############################################################################
# Move and evaluation tables
###############################################################################

N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, 2 * N, N + W, N + E),
    'N': (2 * N + E, N + 2 * E, S + 2 * E, 2 * S + E, 2 * S + W, S + 2 * W, N + 2 * W, 2 * N + W),
    'B': (N + E, S + E, S + W, N + W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N + E, S + E, S + W, N + W),
    'K': (N, E, S, W, N + E, S + E, S + W, N + W)
}

pst = {
    'P': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
          0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
          0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
          0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
          0, 178, 198, 218, 238, 238, 218, 198, 178, 0,
          0, 178, 198, 208, 218, 218, 208, 198, 178, 0,
          0, 178, 198, 198, 198, 198, 198, 198, 178, 0,
          0, 198, 198, 198, 198, 198, 198, 198, 198, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'B': (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 797, 824, 817, 808, 808, 817, 824, 797, 0,
        0, 814, 841, 834, 825, 825, 834, 841, 814, 0,
        0, 818, 845, 838, 829, 829, 838, 845, 818, 0,
        0, 824, 851, 844, 835, 835, 844, 851, 824, 0,
        0, 827, 854, 847, 838, 838, 847, 854, 827, 0,
        0, 826, 853, 846, 837, 837, 846, 853, 826, 0,
        0, 817, 844, 837, 828, 828, 837, 844, 817, 0,
        0, 792, 819, 812, 803, 803, 812, 819, 792, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'N': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 627, 762, 786, 798, 798, 786, 762, 627, 0,
          0, 763, 798, 822, 834, 834, 822, 798, 763, 0,
          0, 817, 852, 876, 888, 888, 876, 852, 817, 0,
          0, 797, 832, 856, 868, 868, 856, 832, 797, 0,
          0, 799, 834, 858, 870, 870, 858, 834, 799, 0,
          0, 758, 793, 817, 829, 829, 817, 793, 758, 0,
          0, 739, 774, 798, 810, 810, 798, 774, 739, 0,
          0, 683, 718, 742, 754, 754, 742, 718, 683, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'R': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 1258, 1263, 1268, 1272, 1272, 1268, 1263, 1258, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'Q': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 2529, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'K': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 60098, 60132, 60073, 60025, 60025, 60073, 60132, 60098, 0,
          0, 60119, 60153, 60094, 60046, 60046, 60094, 60153, 60119, 0,
          0, 60146, 60180, 60121, 60073, 60073, 60121, 60180, 60146, 0,
          0, 60173, 60207, 60148, 60100, 60100, 60148, 60207, 60173, 0,
          0, 60196, 60230, 60171, 60123, 60123, 60171, 60230, 60196, 0,
          0, 60224, 60258, 60199, 60151, 60151, 60199, 60258, 60224, 0,
          0, 60287, 60321, 60262, 60214, 60214, 60262, 60321, 60287, 0,
          0, 60298, 60332, 60273, 60225, 60225, 60273, 60332, 60298, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - End of Sunfish data tables - %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%% - Testing frames are sorted below. For static testing of pre captured images - %%%%%%%%%%%%%%%%%%%%%

# First frame, completely empty board used for detection of squares
empty_board = cv2.imread("\Users\Playtech\Pictures\Logitech Webcam\Picture 70.jpg", 1)

# Filled board in starting config for k means calculations
filled_board = cv2.imread("\Users\Playtech\Pictures\Logitech Webcam\Picture 71.jpg", 1)

# Test frames stating from the first move(not inclusive of board set up and empty board
test_frames = ["\Users\Playtech\Pictures\Logitech Webcam\Picture 72.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 73.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 74.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 75.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 76.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 77.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 78.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 79.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 80.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 81.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 82.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 83.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 84.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 85.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 86.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 87.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 88.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 89.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 90.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 91.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 92.jpg",
               "\Users\Playtech\Pictures\Logitech Webcam\Picture 93.jpg"]


def homography(frame, destination, location, size):
    # Applies homography on a frame in accordance with its location and desired destination

    im_dst = np.zeros(size, np.uint8)

    # Calculate the homography
    h, status = cv2.findHomography(location, destination)

    # Warp source image to destination
    im_dst = cv2.warpPerspective(frame, h, size[0:2])

    return im_dst


def corners_to_list(corners):
    # Extracts the coords from a numpy arrray, places them into tupples and then into a list

    list_corners = corners.tolist()
    tuple_corners = []

    for i in range(0, len(list_corners), 1):
        tuple_corners.append((list_corners[i][0][0], list_corners[i][0][1]))

    return tuple_corners


def order_list(row_count, col_count, list):
    # orders the coords in a way such that G7 = first element then down to G2 and then F,E etc

    all_sorted = []

    sorted_row_y = sorted(list, key=lambda x: x[1])
    for j in range(0, col_count, 1):

        sorted_row_x = sorted(sorted_row_y[(row_count * j):(row_count * (1 + j))], key=lambda x: x[0])

        for i in range(0, row_count, 1):
            all_sorted.append(sorted_row_x[i])

    return all_sorted


def complete_board1(corners):
    # Completes the H and A colums of the board

    # H col
    new_corners = []
    for i in range(0, 7, 1):
        new_corners.append(((corners[i][0]), (corners[i][1] - corners[(7 + i)][1] + corners[i][1])))

    # A col
    for j in range(42, 49, 1):
        new_corners.append(((corners[j][0]), (corners[j][1] - corners[(j - 7)][1] + corners[j][1])))

    for i in range(0, len(new_corners), 1):
        corners.append(new_corners[i])

    return corners


def complete_board2(corners):
    # Completes the rest of the board (col 1 and 8) including the 4 extreem corners

    new_corners = []

    # 8 col
    for k in range(0, 63, 7):
        new_corners.append(((corners[k][0] - corners[((k) + 1)][0] + corners[k][0]), corners[(k)][1]))

    # 1 col
    for k in range(6, 69, 7):
        new_corners.append(((corners[k][0] - corners[((k) - 1)][0] + corners[k][0]), corners[(k)][1]))

    for i in range(0, len(new_corners), 1):
        corners.append(new_corners[i])

    return corners


def BGRtoHSV(color):
    color_np = np.uint8([[color]])
    return cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)


def color_search_range(color, shift1, shift2, shift3):
    lower = []
    upper = []

    #H 0-180
    if (shift1 > color[0]):
        lower.append(0)
    else:
        lower.append(color[0] - shift1)

    if (shift1 + color[0] > 180):
        upper.append(180)
    else:
        upper.append(color[0] + shift1)

    #S 0-255
    if (shift2 > color[1]):
        lower.append(0)
    else:
        lower.append(color[1] - shift2)

    if (shift2 + color[1] > 255):
        upper.append(255)
    else:
        upper.append(color[1] + shift2)

    #V 0-255
    if (shift3 > color[2]):
        lower.append(0)
    else:
        lower.append(color[2] - shift3)

    if (shift3 + color[2] > 255):
        upper.append(255)
    else:
        upper.append(color[2] + shift3)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    return upper,lower


def color_processing(img,hsv, lower, upper):

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(img, img, mask=mask)
    ret,output = cv2.threshold(output,127,255,cv2.THRESH_BINARY)

    blur = cv2.medianBlur(output, 5)

    kernel = np.ones((3, 3), np.uint8)

    #opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    #dilation = cv2.dilate(closing, kernel, iterations=3)
    closed3 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)

    ret, thresh_highlight = cv2.threshold(closed3, 0, 255, cv2.THRESH_BINARY)

    return thresh_highlight


def kmeans_calc(frame):
    Z = frame.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]

    res2 = res.reshape((frame.shape))

    return res2, center


def white_move(diff_white_to, diff_white_from):
    #max_to = max(diff_white_to, key=diff_white_to.get)
    #max_from = max(diff_white_from, key=diff_white_from.get)

    sorted_to = sorted(diff_white_to.items(), key=operator.itemgetter(1))
    sorted_to.reverse()

    sorted_from = sorted(diff_white_from.items(), key=operator.itemgetter(1))
    sorted_from.reverse()

    top_3_to = sorted_to[0:3]
    top_3_from = sorted_from[0:3]

    if ((top_3_to[0][0][1] == top_3_to[1][0][1]) and (top_3_to[1][0][1] == top_3_to[2][0][1])):
        col = [top_3_to[0][0][0], top_3_to[1][0][0], top_3_to[2][0][0]]
        col.sort()

        max_to = col[0] + top_3_from[0][0][1]
    elif (top_3_to[0][0][1] == top_3_to[1][0][1]):
        col = [top_3_to[0][0][0], top_3_to[1][0][0]]
        col.sort()
        max_to = col[0]+top_3_to[0][0][1]

    else:
        max_to = top_3_to[0][0]


    if ((top_3_from[0][0][1] == top_3_from[1][0][1]) and (top_3_from[1][0][1] == top_3_from[2][0][1])):
        col = [top_3_from[0][0][0],top_3_from[1][0][0],top_3_from[2][0][0]]
        col.sort()

        max_from = col[0]+top_3_from[0][0][1]

    elif (top_3_from[0][0][1] == top_3_from[1][0][1]):
        col = [top_3_from[0][0][0], top_3_from[1][0][0]]
        col.sort()
        max_from = col[0]+top_3_from[0][0][1]

    else:
        max_from = top_3_from[0][0]

    return max_from,max_to


def black_move(diff_black_to, diff_black_from):
    #unfinsihed, must copy white move/combine into one without the global frames

    max_to = max(diff_black_to, key=diff_black_to.get)
    max_from = max(diff_black_from, key=diff_black_from.get)

    return max_from,max_to


def capture_frame():
    #unsure why capturing twice removes the delay for images, to do with the read/write buffer??

    ret, frame = cap.read()
    ret, frame = cap.read()

    return frame


# ***************************************** Start up ******************************************************************
def main():

    frame = empty_board
    frame_copy = frame.copy()

    # Applying inbuilt findChessboardCorners,
    found, corners = cv2.findChessboardCorners(frame_copy, (7, 7))
    cv2.drawChessboardCorners(frame_copy, (7, 7), corners, found)


    list_corners = corners_to_list(corners)
    ordered_corners = order_list(7, 7, list_corners)

    # Grabbing the area in which to apply homography to
    top_left = ordered_corners[0]
    top_right = ordered_corners[6]
    bot_left = ordered_corners[42]
    bot_right = ordered_corners[48]

    shifted_top_leftX = int(top_left[0] - (ordered_corners[1][0] - top_left[0]))
    shifted_top_leftY = int(top_left[1] - (ordered_corners[7][1] - top_left[1] + 5))
    shifted_top_rightX = int(top_right[0] - (ordered_corners[5][0] - top_right[0]))
    shifted_top_rightY = int(top_right[1] - (ordered_corners[13][1] - top_right[1] + 5))
    shifted_bot_leftX = int(bot_left[0] - (ordered_corners[36][0] - bot_left[0] + 20))
    shifted_bot_leftY = int(bot_left[1] - (ordered_corners[35][1] - bot_left[1]))
    shifted_bot_rightX = int(bot_right[0] - (ordered_corners[40][0] - bot_right[0] - 20))
    shifted_bot_rightY = int(bot_right[1] - (ordered_corners[41][1] - bot_right[1]))


    # Destination image
    size = (1000, 1000, 3)
    pts_dst = np.array(
        [
            [0, 0],
            [500, 0],
            [500, 500],
            [0, 500]
        ], dtype=float
    )
    # print pts_dst
    pts_src = np.array(
        [
            [shifted_top_leftX, shifted_top_leftY],
            [shifted_top_rightX, shifted_top_rightY],
            [shifted_bot_rightX, shifted_bot_rightY],
            [shifted_bot_leftX, shifted_bot_leftY]
        ], dtype=float
    )

    im_dst = homography(frame, pts_dst, pts_src, size)

    # reapply chess corners on new flat board
    found, corners_homo = cv2.findChessboardCorners(im_dst, (7, 7))

    sorted_corners = corners_to_list(corners_homo)
    square_cords_sorted = order_list(7, 7, sorted_corners)


    completed_board = complete_board1(square_cords_sorted)

    completed_board_half = order_list(7, 9, completed_board)

    complete_board_final = complete_board2(completed_board_half)

    complete_board_final2 = order_list(9, 9, complete_board_final)


    # creating square class


    class Squares:
        def __init__(self, name, topX, topY, botX, botY):
            self.name = name
            self.topX = topX
            self.topY = topY
            self.botX = botX
            self.botY = botY

        def displaySquare(self):
            square_image = im_dst[self.topY:self.botY, self.topX:self.botX]
            cv2.imshow(str(self.name), square_image)

        def squareFrame(self):
            square_frame = im_dst[self.topY:self.botY, self.topX:self.botX]
            return square_frame

        def displayWhite(self):
            white_image = thresh_highlight4[self.topY:self.botY, self.topX:self.botX]
            cv2.imshow(str(self.name + "White"), white_image)

        def displayBlack(self):
            black_image = thresh_highlight3[self.topY:self.botY, self.topX:self.botX]
            cv2.imshow(str(self.name + "Black"), black_image)

        def countWhite(self):
            white_segment = gray_image4[self.topY:self.botY, self.topX:self.botX]
            nzCount_white = np.count_nonzero(white_segment)
            return nzCount_white

        def countBlack(self):
            black_segment = gray_image3[self.topY:self.botY, self.topX:self.botX]
            nzCount_black = np.count_nonzero(black_segment)
            return nzCount_black

        def countWhite_to(self):
            segment = difference_white_to[self.topY:self.botY, self.topX:self.botX]
            return np.count_nonzero(segment)

        def countWhite_from(self):
            segment = difference_white_from[self.topY:self.botY, self.topX:self.botX]
            return np.count_nonzero(segment)

        def countBlack_to(self):
            segment = difference_black_to[self.topY:self.botY, self.topX:self.botX]
            return np.count_nonzero(segment)

        def countBlack_from(self):
            segment = difference_black_from[self.topY:self.botY, self.topX:self.botX]
            return np.count_nonzero(segment)




    # DICTS to populate all of the row g (g2-g7)
    square_cords = dict()
    row_group = dict()

    diff_white_to = dict()
    diff_white_from = dict()
    diff_black_to = dict()
    diff_black_from = dict()

    for j in range(0, 8, 1):
        for i in range(0, 8, 1):
            square_cords['%c%dx1' % ((72 - j), (8 - i))] = int(complete_board_final2[i + j * 9][0])
            square_cords['%c%dy1' % ((72 - j), (8 - i))] = int(complete_board_final2[i + j * 9][1])
            square_cords['%c%dx2' % ((72 - j), (8 - i))] = int(complete_board_final2[i + 10 + j * 9][0])
            square_cords['%c%dy2' % ((72 - j), (8 - i))] = int(complete_board_final2[i + 10 + j * 9][1])

            row_group['%c%d' % ((72 - j), (8 - i))] = Squares("%c%d" % ((72 - j), (8 - i)),
                                                              square_cords['%c%dx1' % ((72 - j), (8 - i))],
                                                              square_cords['%c%dy1' % ((72 - j), (8 - i))],
                                                              square_cords['%c%dx2' % ((72 - j), (8 - i))],
                                                              square_cords['%c%dy2' % ((72 - j), (8 - i))])

            diff_white_to['%c%d' % ((72 - j), (8 - i))] = 0
            diff_white_from['%c%d' % ((72 - j), (8 - i))] = 0
            diff_black_to['%c%d' % ((72 - j), (8 - i))] = 0
            diff_black_from['%c%d' % ((72 - j), (8 - i))] = 0

    # for i in range(1,9,1):


    # Live homography destination @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    size = (1000, 1000, 3)
    pts_dst = np.array(
        [
            [0, 0],
            [500, 0],
            [500, 500],
            [0, 500]
        ], dtype=float
    )
    pts_src = np.array(
        [
            [shifted_top_leftX, shifted_top_leftY],
            [shifted_top_rightX, shifted_top_rightY],
            [shifted_bot_rightX, shifted_bot_rightY],
            [shifted_bot_leftX, shifted_bot_leftY]
        ], dtype=float
    )

    im_src = filled_board

    im_dst = homography(im_src, pts_dst, pts_src, size)
    im_dst_copy = im_dst.copy()

    im_crop = im_dst_copy[int(complete_board_final2[0][1]):int(complete_board_final2[80][1]),
              int(complete_board_final2[0][0]):int(complete_board_final2[80][0])]

    hsv = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)

    res2,center = kmeans_calc(im_crop)

    #cv2.imshow('res2', res2)
    cv2.waitKey()

    #PROCESSING OF IMAGE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    color_1 = BGRtoHSV(center[0])[0][0]
    color_2 = BGRtoHSV(center[3])[0][0]
    color_3 = BGRtoHSV(center[2])[0][0] #black
    color_4 = BGRtoHSV(center[1])[0][0] #white


    upper1,lower1 = color_search_range(color_1,30,30,30)
    upper2,lower2 = color_search_range(color_2,30,60,30)
    upper3,lower3 = color_search_range(color_3,20,20,100)
    upper4,lower4 = color_search_range(color_4,45,45,30)

    thresh_highlight1 = color_processing(im_crop,hsv,lower1,upper1)
    thresh_highlight2 = color_processing(im_crop,hsv,lower2,upper2)
    thresh_highlight3 = color_processing(im_crop,hsv,lower3,upper3)
    thresh_highlight4 = color_processing(im_crop,hsv,lower4,upper4)

    #cv2.imshow("image MASK 3_4", np.hstack([im_crop, thresh_highlight3, thresh_highlight4]))

    gray_image3 = cv2.cvtColor(thresh_highlight3, cv2.COLOR_BGR2GRAY)
    gray_image4 = cv2.cvtColor(thresh_highlight4, cv2.COLOR_BGR2GRAY)




    ##print "White"
   # print row_group["E7"].countWhite()
    #print "Black"
    #print row_group["E7"].countBlack()


    # ############################################ KERNAL LOCATION ######################################################
    count = 0
    turn = "W"

    pos = Position(initial, 0, (True, True), (True, True), 0, 0)

    while (1):


        previous_black = gray_image3
        previous_white = gray_image4

        # Homo
        # Read in the image.
        im_src = cv2.imread(test_frames[count])                   # <<<<<<<< RE ENABLE FOR PROPER TESTING (EXTRACTING FROM THE IMAGES)


        #$$$$$$$$ CHANGE FOR LIVE FEED OR IMAGE
        #ret, im_src = cap.read()
        #im_src = cv2.imread("\Users\Playtech\Pictures\Logitech Webcam\Picture 2.jpg", 1)

        # Destination image


        im_dst = homography(im_src, pts_dst, pts_src, size)

        im_dst_copy = im_dst.copy()

        im_crop = im_dst_copy[int(complete_board_final2[0][1]):int(complete_board_final2[80][1]),
                  int(complete_board_final2[0][0]):int(complete_board_final2[80][0])]

        #for j in range(0,8,1):
        #    for i in range (0,8,1):
        #         row_group['%c%d' % ((72 - j), (8 - i))].displaySquare()

        hsv = cv2.cvtColor(im_crop, cv2.COLOR_BGR2HSV)


        #cv2.imshow('res2', res2)

        # PROCESSING OF IMAGE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        thresh_highlight1 = color_processing(im_crop, hsv, lower1, upper1)
        thresh_highlight2 = color_processing(im_crop, hsv, lower2, upper2)
        thresh_highlight3 = color_processing(im_crop, hsv, lower3, upper3)
        thresh_highlight4 = color_processing(im_crop, hsv, lower4, upper4)

        #cv2.imshow("image MASK 3_4", np.hstack([im_crop,thresh_highlight3, thresh_highlight4]))

        gray_image3 = cv2.cvtColor(thresh_highlight3, cv2.COLOR_BGR2GRAY)
        gray_image4 = cv2.cvtColor(thresh_highlight4, cv2.COLOR_BGR2GRAY)

        difference_black_both = cv2.absdiff(gray_image3,previous_black)
        difference_black_from = cv2.medianBlur(cv2.bitwise_and(previous_black, previous_black, mask=difference_black_both),5)
        difference_black_to = cv2.medianBlur(cv2.bitwise_and(gray_image3, gray_image3, mask=difference_black_both),5)

        difference_white_both = cv2.absdiff(gray_image4,previous_white)
        difference_white_from = cv2.medianBlur(cv2.bitwise_and(previous_white, previous_white, mask=difference_white_both),5)
        difference_white_to = cv2.medianBlur(cv2.bitwise_and(gray_image4, gray_image4, mask=difference_white_both),5)

        #cv2.imshow("Diff Black", np.hstack([difference_black_both,difference_black_to, difference_black_from]))
        cv2.imshow("Diff White", np.hstack([difference_white_both,difference_white_to, difference_white_from]))
        cv2.imshow("Main", im_crop)
        #cv2.waitKey()
        for j in range(0,8,1):
            for i in range(0,8,1):
                diff_white_to['%c%d' % ((72 - j), (8 - i))] = row_group['%c%d' % ((72 - j), (8 - i))].countWhite_to()
                diff_white_from['%c%d' % ((72 - j), (8 - i))] = row_group['%c%d' % ((72 - j), (8 - i))].countWhite_from()
                diff_black_to['%c%d' % ((72 - j), (8 - i))] = row_group['%c%d' % ((72 - j), (8 - i))].countBlack_to()
                diff_black_from['%c%d' % ((72 - j), (8 - i))] = row_group['%c%d' % ((72 - j), (8 - i))].countBlack_from()

        my_move = (white_move(diff_white_to, diff_white_from))
        my_move = str.lower(str(my_move[0]) + str(my_move[1]))

        #moving_test = move_list[count] # TEST AS ABOVE IS INVALID <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        moving_test = my_move

        count += 2

        print("Press Enter once you have made your move...")
        cv2.waitKey()


        print "Detected move: " + moving_test


        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k == 32:
            square_frame = row_group["G2"].squareFrame()
            cv2.imshow("G2 Snapshot", square_frame)
        elif k == 122:
            new_square_frame = row_group["G2"].squareFrame()
            diff = cv2.subtract(new_square_frame, square_frame)
            cv2.imshow("G2 diff", diff)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    #cv2.destroyAllWindows()
    #cap.release()


main()
