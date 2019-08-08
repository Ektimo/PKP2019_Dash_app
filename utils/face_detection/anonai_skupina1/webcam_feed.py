import cv2
import FaceBlurUtils as utls
from FaceDetector import FaceDetector
from paths.paths import model_path

useDetectionEnhancement = True

camFeed = cv2.VideoCapture(0)
if camFeed.isOpened():
    ret, frame = camFeed.read()
else:
    print("Failed opening the camera feed")
    exit(-1)

video_xdim = frame.shape[0]
video_ydim = frame.shape[1]
print("camera resolution: ", video_xdim, video_ydim)

video_path_output = 'live_feed_output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(video_path_output, fourcc, 20.0, (640,480))

detector = FaceDetector(model_path=model_path, face_tracking="kalman", max_frame_padding=10)

while camFeed.isOpened():
    ret, frame = camFeed.read()

    if ret == True:
        if useDetectionEnhancement:
            res, result_container = detector.face_blur(frame)

            if res == detector.RESULT_FRAME:
                output_video.write(result_container)
                cv2.imshow('frame', result_container)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if res == detector.RESULT_QUEUE:
                print("returned cache size ", len(result_container))
                for i in range(0, len(result_container)):
                    blurred = result_container.pop()
                    output_video.write(blurred)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        else:
            blurred = detector.blur_single_image(frame)
            output_video.write(blurred)

            cv2.imshow('frame', blurred)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        if useDetectionEnhancement:
            # empty the current detector cache
            for i in range(0, len(detector.frame_cache)):
                output_video.write(detector.frame_cache.pop())
        break

output_video.release()
cv2.destroyAllWindows()

input_video = cv2.VideoCapture(video_path_output)
utls.PlayVideo(input_video)

input_video.release()
cv2.destroyAllWindows()