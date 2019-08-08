import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path_1 = "videos/output_gilmore_girls_a_second_film_by_kirk.avi"
video_path_2 = "videos/output_gilmore_girls_a_second_film_by_kirk.avi"

video_paths = [video_path_1, video_path_2]
windows_titles = [str(i) for i in range(0, len(video_paths))]

video_captures = [cv2.VideoCapture(i) for i in video_paths]

frames = [None] * len(video_captures)
gray = [None] * len(video_captures)
ret = [None] * len(video_captures)

print(frames, gray, ret)


def PlayVideo(input_video):
    frame_count = 0
    while input_video.isOpened():
        frame_count = frame_count + 1
        ret, frame = input_video.read()
        if ret is True:
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    print("end of video frame count:", frame_count)


#PlayVideo(video_captures[1])
#PlayVideo(video_captures[1])

while True:

    for i, c in enumerate(video_captures):
        if c is not None:
            ret[i], frames[i] = c.read()

    for i in range(0, len(ret)):
        if ret[i] is True:
            cv2.imshow(windows_titles[i], frames[i])

    if cv2.waitKey(50000000) & 0xFF == ord('q'):
        break

for cap in video_captures:
    cap.release()

cv2.destroyAllWindows()
