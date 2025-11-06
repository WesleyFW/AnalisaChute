import mediapipe as mp
from mediapipe.tasks import python
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import threading
import queue
import math



class AnalisaChutes:
    def __init__(self, file_name= ''):
        self.running = True
        self.paused = False
        self.file_name = file_name
        self.image_q = queue.Queue()
        moddel_path = 'pose_landmarker_full.task'

        self.options = python.vision.PoseLandmarkerOptions(
          base_options=python.BaseOptions(model_asset_path=moddel_path),
          running_mode=python.vision.RunningMode.VIDEO)

    def find_angle(self, frame, landmarks, p1, p2 ,p3, draw):
        land = landmarks.pose_landmarks[0]
        h,w,c = frame.shape
        x1, y1=(land[p1].x, land[p1].y)
        x2, y2=(land[p2].x, land[p2].y)
        x3, y3=(land[p3].x, land[p3].y)

        angle = math.degrees(math.atan2(y3-y2, x3-x2) -
                                math.atan2(y1-y2,x1-x2))
        position = (int(x2*w+10), int(y2*h+10))
        if draw:
            frame = cv2.putText(frame, str(int(angle)), position,
                           cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,0), 2)
            
        return frame, angle

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def process_video(self, draw , display):

        with python.vision.PoseLandmarker.create_from_options(self.options) as landmarker: 
            cap = cv2.VideoCapture(self.file_name)
            calc_ts = [0.0]
           
            while cap.isOpened() and self.running:
                if self.paused:
                    cv2.waitKey(50)
                    continue 

                ret, frame = cap.read()
                if not ret:
                    break

                ret, frame = cap.read()
                fps = cap.get(cv2.CAP_PROP_FPS)

                if ret == True:
                    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
                    calc_ts.append(int(calc_ts[-1] + 1000/fps))
                    detection_result = landmarker.detect_for_video(mp_image, calc_ts[-1])

                    if draw:
                        frame = self.draw_landmarks_on_image(frame, detection_result)
                        
                    if display:
                        cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) &  0xFF == ord('q'):
                          break
                    if  0xFF == ord('p'):
                            cv2.waitKey(-1)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    self.image_q.put((frame, detection_result, calc_ts[-1]))
                else:
                    break
        self.image_q(1,1, "Finalizado")           
        cap.release()
        cv2.destroyAllWindows()

    
    def run(self, draw, display = False):
        self.running = True
        t1 = threading.Thread(target=self.process_video, args=(draw,display))
        t1.start()
    
    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

if __name__ == "__main__":
    analisachutes = AnalisaChutes()
    analisachutes.process_video(False, True)