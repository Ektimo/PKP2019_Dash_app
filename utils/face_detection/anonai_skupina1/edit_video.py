import cv2
from FaceDetector import FaceDetector
from paths.paths import model_path, image_path, video_path
import time
import FaceBlurUtils as utls

start_time = time.time()

print("opening: ", video_path)
input_video = cv2.VideoCapture(video_path)
video_xdim = 800
video_ydim = 600

if input_video.isOpened():
    video_xdim = int(input_video.get(3))
    video_ydim = int(input_video.get(4))
    print('video dimensions: ', video_xdim, 'x', video_ydim)
else:
    print("failed to open video file")
    exit(-1)


#PlayVideo(input_video)
#exit(0)

# specify video encoding and format
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_path_output = video_path.split("/")
video_path_output[-1] = "output_" + video_path_output[-1].split(".")[0]
video_path_output = "/".join(video_path_output) + ".avi"

output_video = cv2.VideoWriter(video_path_output, fourcc, 20.0, (video_xdim, video_ydim))
print("output to: ", video_path_output)

# init face detection model
# face_detector = fd.detection_model_init(model_path)
detector = FaceDetector(model_path=model_path, face_tracking="kalman", max_frame_padding=10)
cntr = 0

while(input_video.isOpened()):
    cntr = cntr + 1
    print(" frame: ", cntr)
    # ret is False if there are no more frames to read
    ret, frame = input_video.read()
    if ret == True:

        # do stuff with frame
        res, result_container = detector.face_blur(frame)

        if res == detector.RESULT_FRAME:
            output_video.write(result_container)
        
        if res == detector.RESULT_QUEUE:
            # print("interpolating: ", detector.face_coordinates_falling, detector.face_coordinates_rising, "delta: ", detector.face_coordinates_delta)
            print("returned cache size ", len(result_container))
            for i in range(0, len(result_container)):
                output_video.write(result_container.pop())
            
    else:
        # empty the current detector cache
        for i in range(0, len(detector.frame_cache)):
            output_video.write(detector.frame_cache.pop())

        print("write done")
        break
    
input_video.release()
output_video.release()
cv2.destroyAllWindows()

end_time = time.time()
print(end_time - start_time)

input_video = cv2.VideoCapture(video_path_output)
utls.PlayVideo(input_video)

input_video.release()
cv2.destroyAllWindows()
