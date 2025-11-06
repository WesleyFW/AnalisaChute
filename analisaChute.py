import mediapipe as mp
from mediapipe.tasks import python
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import threading
import queue
import math
import time


class AnalisaChutes:
    def __init__(self, file_name=''):
        self.running = False
        self.paused = False
        self.file_name = file_name
        self.image_q = queue.Queue()
        model_path = 'pose_landmarker_full.task'

        self.options = python.vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=python.vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Thread control
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

    def find_angle(self, frame, landmarks, p1, p2, p3, draw=True, person_idx=0):
        if landmarks is None or not hasattr(landmarks, "pose_landmarks"):
            return frame, None

        if len(landmarks.pose_landmarks) <= person_idx:
            return frame, None

        land = landmarks.pose_landmarks[person_idx]
        h, w = frame.shape[:2]

        x1, y1 = (land[p1].x, land[p1].y)
        x2, y2 = (land[p2].x, land[p2].y)
        x3, y3 = (land[p3].x, land[p3].y)

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        position = (int(x2 * w + 10), int(y2 * h + 10))

        if draw:
            cv2.putText(frame, str(int(angle)), position,
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        return frame, angle

    def draw_landmarks_on_image(self, rgb_image, detection_result, draw=False):
        pose_landmarks_list = getattr(detection_result, "pose_landmarks", [])
        annotated_image = np.copy(rgb_image)
        all_people = []

        for pose_landmarks in pose_landmarks_list:
            person_points = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
            all_people.append(person_points)

            if draw:
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())

        return annotated_image, all_people

    def process_video(self, draw=False, display=False):
        self._stop_event.clear()
        self._pause_event.clear()
        self.running = True

        try:
            with python.vision.PoseLandmarker.create_from_options(self.options) as landmarker:
                cap = cv2.VideoCapture(self.file_name)
                calc_ts = [0]

                while cap.isOpened() and not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        time.sleep(0.05)
                        continue

                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    calc_ts.append(int(calc_ts[-1] + 1000 / fps))
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    detection_result = landmarker.detect_for_video(mp_image, calc_ts[-1])

                    pose_landmarks_list = getattr(detection_result, "pose_landmarks", [])
                    all_people = []
                    annotated_frame = frame_rgb
                    if draw:
                        annotated_frame = np.copy(frame_rgb)

                    for pose_landmarks in pose_landmarks_list:
                        person_points = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks])
                        all_people.append(person_points)
                        if draw:
                            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                            pose_landmarks_proto.landmark.extend([
                                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                                for landmark in pose_landmarks
                            ])
                            solutions.drawing_utils.draw_landmarks(
                                annotated_frame,
                                pose_landmarks_proto,
                                solutions.pose.POSE_CONNECTIONS,
                                solutions.drawing_styles.get_default_pose_landmarks_style())

                    out_frame = annotated_frame if draw else frame_rgb

                    if display:
                        show_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Frame', show_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    self.image_q.put((out_frame, detection_result, all_people, calc_ts[-1]))

                self.image_q.put(("END", None, None, None))
                cap.release()
                if display:
                    cv2.destroyAllWindows()
        finally:    
            self.running = False
            self._stop_event.clear()
            self._pause_event.clear()

    def run(self, draw=False, display=False, restart=False):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if restart:
                    self.stop()
                else:
                    return
            self._stop_event.clear()
            self._pause_event.clear()
            self.running = True
            self._thread = threading.Thread(target=self.process_video, args=(draw, display), daemon=True)
            self._thread.start()
    def pause(self):
        self._pause_event.set()
        self.paused = True
    def resume(self):
        self._pause_event.clear()
        self.paused = False
    def stop(self, timeout=2.0):
        with self._lock:
            self._stop_event.set()
            self._pause_event.clear()
            if self._thread is not None:
                self._thread.join(timeout)
                if self._thread.is_alive():
                    try:
                        while not self.image_q.empty():
                            self.image_q.get_nowait()
                    except Exception:
                        pass
                    return False
                self._thread = None
            try:
                while not self.image_q.empty():
                    self.image_q.get_nowait()
            except Exception:
                pass
            self.running = False
            self.paused = False
        return True



if __name__ == "__main__":
    analisachutes = AnalisaChutes("video.mp4")
    analisachutes.run(draw=False, display=False)