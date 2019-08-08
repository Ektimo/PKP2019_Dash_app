import cv2
import numpy as np
import InsightfaceExample as fd
from collections import deque
from KalmanFilter import KalmanFilter
import FaceBlurUtils as utls

class FaceDetector:
    RESULT_NONE = 0
    RESULT_FRAME = 1
    RESULT_QUEUE = 2

    NO_EDGE = 0
    EDGE_FALLING = 1
    EDGE_RISING = 2

    NO_TRACKING = 0
    TRACKING_QUEUE = 1
    TRACKING_KALMAN = 2

    FILTER_MATCH = 0
    FILTER_NO_MATCH = -1

    def __init__ (self, model_path, mtcnn_path=None, frame_size=(800, 600), face_tracking="none", max_frame_padding=10, frame_crop=False, frame_crop_ratio = 0.1):
        self.max_frame_padding = max_frame_padding
        self.detection_model = fd.detection_model_init(model_path, mtcnn_path=mtcnn_path)

        self.face_tracking = self.NO_TRACKING
        if face_tracking is "queue":
            self.face_tracking = self.TRACKING_QUEUE
        elif face_tracking is "kalman":
            self.face_tracking = self.TRACKING_KALMAN

        # queue face tracking
        # rectangle: [x, y, w, h]
        self.face_coordinates_falling = [10, 10, 500, 500]
        self.face_coordinates_rising = [10, 10, 500, 500]
        self.face_coordinates_delta = [0, 0, 0, 0]
        self.prev_frame_has_face = True

        self.use_frame_cache = False
        self.frame_cache = deque()

        # kalman filter face tracking
        self.k_filters = []

        # list of filters with detected face in current iteration 
        self.k_filters_with_face = np.zeros(len(self.k_filters), dtype=np.bool)


    def __edge_detect(self, current):
        # @brief detects edge between previus and current face detection result
        result = self.NO_EDGE

        if (self.prev_frame_has_face == False) and (current == True):
            result = self.EDGE_RISING
        if (self.prev_frame_has_face == True) and (current == False):
            result = self.EDGE_FALLING
        
        self.prev_frame_has_face = current

        return result


    def __delta_coordinates(self):
        # @brief computes embedding rectangle for face rectangles before falling and after rising edge

        # encapsulate rectangle before falling edge and rec after rising edge
        rec_x = min(self.face_coordinates_rising[0], self.face_coordinates_falling[0])
        rec_y = min(self.face_coordinates_rising[1], self.face_coordinates_falling[1])
        rec_w = max(self.face_coordinates_rising[2], self.face_coordinates_falling[2]) + \
                abs(self.face_coordinates_rising[0] - self.face_coordinates_falling[0])
        rec_h = max(self.face_coordinates_rising[3], self.face_coordinates_falling[3]) + \
                abs(self.face_coordinates_rising[1] - self.face_coordinates_falling[1])

        rec = [rec_x, rec_y, rec_w, rec_h]
        return rec


    def __find_proxy_filter(self, face_box):
        # @brief determines whether the center of the face_box fits in any of the kalman filters

        face_box_center = utls.rectangle_center(face_box)
        for i, filter in enumerate(self.k_filters):
            if filter.has_in_rectangle(face_box_center):
                return self.FILTER_MATCH, i
            
        return self.FILTER_NO_MATCH, None


    def __delete_kalman_filters(self, indexes):
        # @brief deletes Kalman filters by their index

        self.k_filters = np.delete(self.k_filters, indexes)
        self.k_filters_with_face = np.delete(self.k_filters_with_face, indexes)


    def blur_single_image(self, image):
        # @brief detects and blurs the faces in a single image
        res, image_blurred, face_boxes = fd.detect_face_video_frame(image, self.detection_model)
        return image_blurred

    def face_blur(self, frame):
        # face_boxes are null if res is not 0
        res, frame_blurred, face_boxes = fd.detect_face_video_frame(frame, self.detection_model)

        # if frame_crop:
        # TODO

        if self.face_tracking == self.TRACKING_QUEUE:

            frame_has_face = False
            if res == 0:
                frame_has_face = True

            edge = self.__edge_detect(frame_has_face)
            
            if edge == self.EDGE_RISING:
                if self.use_frame_cache == True:
                    # TODO multiple faces can be detected
                    self.face_coordinates_rising = [face_boxes[0][0], face_boxes[0][1], face_boxes[0][2], face_boxes[0][3]]
                    self.face_coordinates_delta = self.__delta_coordinates()

                    # interpolate and return
                    retcache = deque()
                    for i in range(0, len(self.frame_cache)):
                        res_img = fd.blur_face(self.frame_cache.pop(), \
                                            self.face_coordinates_delta[0], \
                                            self.face_coordinates_delta[1], \
                                            self.face_coordinates_delta[2], \
                                            self.face_coordinates_delta[3])
                        retcache.appendleft(res_img)

                    self.use_frame_cache = False
                    return self.RESULT_QUEUE, retcache
                else:
                    return self.RESULT_FRAME, frame_blurred

            elif edge == self.EDGE_FALLING:
                self.frame_cache.clear()
                self.use_frame_cache = True

            if face_boxes is not None:
                # TODO multiple faces can be detected
                self.face_coordinates_falling = [face_boxes[0][0], face_boxes[0][1], face_boxes[0][2], face_boxes[0][3]]

            if self.use_frame_cache == True:
                print("cached")
                self.frame_cache.appendleft(frame_blurred)
                if len(self.frame_cache) >= self.max_frame_padding:
                    self.use_frame_cache = False
                    return self.RESULT_QUEUE, self.frame_cache
                else:
                    return self.RESULT_NONE, None
            
            else:
                return self.RESULT_FRAME, frame_blurred
        
        elif self.face_tracking == self.TRACKING_KALMAN:

            self.k_filters_with_face.fill(False)

            if res == 0:
                # assign face boxes to filters, if a new box appears create a new filter
                for i, face_box in enumerate(face_boxes):

                    res, filter_idx = self.__find_proxy_filter(face_box)
                    if res == self.FILTER_MATCH:
                        # assign filter to box, update filter and predict the next face location 
                        self.k_filters_with_face[filter_idx] = True
                        k_filter = self.k_filters[filter_idx]
                        k_filter.update_and_correct(face_box)
                        k_filter.predict()
                    else:
                        # initialize a new kalman filter
                        k_filter = KalmanFilter(init_rectangle=face_box)
                        self.k_filters = np.append(self.k_filters, k_filter)
                        self.k_filters_with_face = np.append(self.k_filters_with_face, True)

            delete_filters = []

            # if any filter has no face box assigned use filter prediction
            for i, status in enumerate(self.k_filters_with_face):
                if status == False:
                    k_filter = self.k_filters[i]

                    # mark filter for deletion if padding limit has been achieved
                    if k_filter.n_padded_frames >= self.max_frame_padding:
                        delete_filters.append(i)
                        continue

                    k_prediction = k_filter.prediction
                    k_prediction = np.squeeze(k_prediction.astype(int))
                    rul_x, rul_y = utls.rectangle_upper_left([k_prediction[0], \
                                                            k_prediction[1], \
                                                            k_filter.rec_width, \
                                                            k_filter.rec_height])
                    print("upper left: ", rul_x, rul_y, k_filter.rec_width, k_filter.rec_height)

                    # TODO change blur_face to accept rectangle as a list
                    frame_blurred = fd.blur_face(frame_blurred, \
                                                rul_x, \
                                                rul_y, \
                                                k_filter.rec_width, \
                                                k_filter.rec_height)

                    frame_blurred = cv2.circle(frame_blurred, (k_prediction[0], k_prediction[1]), 5, (0, 0, 255), -1)
                    frame_blurred = cv2.putText(frame_blurred, str(len(self.k_filters)), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    frame_blurred = cv2.putText(frame_blurred, str(self.k_filters_with_face), (500, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                    k_filter.predict()
                    k_filter.n_padded_frames += 1

            self.__delete_kalman_filters(delete_filters)
            return self.RESULT_FRAME, frame_blurred

        else:
            return self.RESULT_FRAME, frame_blurred


    def blur_video(self, video_path):
        # @brief blurs the video file given the url to it
        # the resulting video file will be located in the same directory with output_ prepended to its name

        print("opening video file: ", video_path)
        input_video = cv2.VideoCapture(video_path)

        video_xdim = 800
        video_ydim = 600

        if input_video.isOpened():
            video_xdim = int(input_video.get(3))
            video_ydim = int(input_video.get(4))
            print('video dimensions: ', video_xdim, 'x', video_ydim)
        else:
            print("failed to open video file")
            return
        
        # specify video encoding and format
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        video_path_output = video_path.split("/")
        video_path_output[-1] = "blurred_" + video_path_output[-1]
        video_path_output = "/".join(video_path_output)

        output_video = cv2.VideoWriter(video_path_output, fourcc, 20.0, (video_xdim, video_ydim))
        print("output to: ", video_path_output)

        while input_video.isOpened():
            ret, frame = input_video.read()
            if ret == True:
                res, result_container = self.face_blur(frame)

                if res == self.RESULT_FRAME:
                    output_video.write(result_container)
                
                if res == self.RESULT_QUEUE:
                    for i in range(0, len(result_container)):
                        output_video.write(result_container.pop())
            
            else:
                # empty the current detector cache
                for i in range(0, len(self.frame_cache)):
                    output_video.write(self.frame_cache.pop())
                break

        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()
        print("write done")

