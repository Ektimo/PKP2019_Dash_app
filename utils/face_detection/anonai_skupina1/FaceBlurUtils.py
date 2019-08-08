import cv2

def rectangle_center(rectangle):
    # @brief computes the center point for the input rectangle given as [x_upper_left, y_upper_left, width, height]
    
    c_x = rectangle[0] + round(rectangle[2] / 2)
    c_y = rectangle[1] + round(rectangle[3] / 2)
    return [c_x, c_y]


def rectangle_upper_left(rectangle):
    # @brief converts rectangle specified as [center_x, center_y, width, height] into upper-left-corner format
    x = round(rectangle[0] - rectangle[2]/2)
    y = round(rectangle[1] - rectangle[3]/2)
    return [int(x), int(y)]


def PlayVideo(input_video, frame_wait_period=30):
    # @brief plays input video, which should already be loaded by opencv

    frame_count = 0

    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret is True:
            frame_count = frame_count + 1
            cv2.imshow('frame', frame)
            if cv2.waitKey(frame_wait_period) & 0xFF == ord('q'):
                break
        else:
            break
    print("end of video frame count:", frame_count)
