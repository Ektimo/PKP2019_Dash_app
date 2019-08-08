import cv2
import numpy as np
import FaceBlurUtils as utls

class KalmanFilter:
    def __init__ (self, init_rectangle = [0, 0, 0, 0]):
        # @param init_rectangle: [x,y,w,h]

        self.n_padded_frames = 0
        self.rec_width = init_rectangle[2]
        self.rec_height = init_rectangle[3]

        # constant speed model
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        self.kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                                [0., 1., 0., .1], 
                                                [0., 0., 1., 0.],
                                                [0., 0., 0., 1.]])
        self.kalman.measurementMatrix = 1. * np.eye(2, 4)      # TODO you can tweak these to make the tracker
        self.kalman.processNoiseCov = 1e-5 * np.eye(4, 4)      # respond faster to change and be less smooth
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
        self.kalman.errorCovPost = 1e-1 * np.eye(4, 4)

        c_x, c_y = utls.rectangle_center(init_rectangle)
        init_kalman_state = np.array([c_x, c_y, 0, 0], dtype='float64')
        self.kalman.statePost = init_kalman_state

        # [c_x, c_y, v_x, v_y]
        self.prediction = self.kalman.predict()


    def predict(self):
        # @brief predicts the next face location given the current measurement

        self.prediction = self.kalman.predict()
        return self.prediction


    def update_and_correct(self, measurement):
        # @brief updates the kalman filter parameters with measurement [x,y,w,h]

        self.rec_height = measurement[2]
        self.rec_width = measurement[3]
        c_x, c_y = utls.rectangle_center(measurement)
        measurement = np.matrix(np.array([float(c_x), float(c_y)], dtype='float64')).transpose()
        self.kalman.correct(measurement)


    def has_in_rectangle(self, point):
        # @brief returns True if the point [c_x, c_y] is inside the rectangle given as [x_upper_left, y_upper_left, width, height]

        rec_x, rec_y = utls.rectangle_upper_left([int(self.prediction[0]), \
                                                int(self.prediction[1]), \
                                                self.rec_width, \
                                                self.rec_height])

        if (point[0] >= rec_x) and \
           (point[0] <= (rec_x + self.rec_width)) and \
           (point[1] >= rec_y) and \
           (point[1] <= (rec_y + self.rec_height)):
            return True
        else:
            return False
