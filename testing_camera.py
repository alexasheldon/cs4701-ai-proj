import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

CAM_INDEX = 0
WINDOW_NAME = "MediaPipe Hands - press q to quit, s to save"

# default threshold for detecting gesture start / end
START_THRESH = 0.015
END_THRESH = 0.010 # originally 0.010 (less than START THRESH)
DEBOUNCE_START = 3
DEBOUNCE_END = 3 # originally 8
SMOOTH_WINDOW = 7 # originally 5
MIN_SEG_FRAMES = 6

CAL_INTERVAL = 5  # seconds between calibration prompts
MOTION_HISTORY_MAX = 3000 # max length of motion history deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Normalize landmarks: translate so wrist is at origin, scale to fit in unit sphere
def normalize_landmarks(landmarks):
    # handle empty landmarks
    if not landmarks or len(landmarks.landmark) == 0:
        return np.empty((0,3))
    # Convert landmarks to numpy array
    lm = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark])

    # Translate landmarks so that wrist is at origin
    lm -= lm[0]
    # Calculate scale (max distance from origin)
    scale = np.linalg.norm(lm[9]) if np.linalg.norm(lm[9]) > 1e-6 else np.max(np.linalg.norm(lm, axis=1))
    lm /= (scale + 1e-8)
    return lm

# landmarks flattened into feature vector
def flatten(lm):
    return lm.flatten()

# compute motion energy between two sets of landmarks
def motion_energy(curr_lm, prev_lm):
    if curr_lm.ndim != 2 or prev_lm.ndim != 2:
        print("Invalid input to motion_energy: curr_lm or prev_lm is not 2D")
        return 0  # Return zero motion energy for invalid landmarks
    return np.mean(np.linalg.norm(curr_lm - prev_lm, axis=1))

# def draw_indicator(frame, text, color=(0,200,0)):
#     h, w = frame.shape[:2]
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0,0), (w,40), color, -1)
#     alpha = 0.35
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#     cv2.putText(frame, text, (10, 34),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    
# # convert seconds and fps to frame count
# def secs_to_frames(s, fps):
#     return max(1, int(round(s * fps)))

# # suggest thresholds based on motion history
# def suggest_thresholds(motion_vals):
#     if len(motion_vals)<30:
#         return None
#     motion_vals_arr = np.array(motion_vals)
#     p50 = np.percentile(motion_vals_arr, 50)
#     p90 = np.percentile(motion_vals_arr, 90)
#     # could add more percentiles
#     start_thresh = p90
#     end_thresh = p50
#     return {"p50" : p50, "p90" : p90, "start_thresh": start_thresh, "end_thresh": end_thresh}

letter_mapping = {
        0 :'A',
        1 :'B',
        2 :'C',
        3 :'D',
        4 :'E',
        5 :'F',
        6 :'G',
        7 :'H',
        8 :'I',
        9 :'K',
        10 :'L',
        11 :'M',
        12 :'N',
        13 :'O',
        14 :'P',
        15 :'Q',
        16 :'R',
        17 :'S',
        18 :'T',
        19 :'U',
        20 :'V',
        21 :'W',
        22 :'X',
        23 :'Y'
    }

def main():
    global START_THRESH, END_THRESH, DEBOUNCE_START, DEBOUNCE_END # make editable later

    # Open with AVFoundation to use Continuity Camera (iPhone)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)

    # camera init wait
    time.sleep(0.2)

    if not cap.isOpened():
        print("Camera not accessible. Check macOS Camera permissions and index.")
        return

    # Setting a resolution to stabilize negotiation (my resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # saved = 0
    # prev_time = time.time()
    # frames = 0

    # state for debounce and motion
    prev_lm = None
    motion_buf = deque(maxlen=SMOOTH_WINDOW)
    motion_history = deque(maxlen=MOTION_HISTORY_MAX)
    in_segment = False
    seg_frames = []
    frame_idx = 0
    start_counter = 0
    end_counter = 0
    frames = 0
    prev_time = 0
    num_signs = 0
    letter = ""

    # Load model
    model = torch.load("mlp_model_norm.pt")

    # Mediapipe Hands setup
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2, # maybe 1?
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        try:
            while True:
                # fetching frame into buffer
                if not cap.grab():
                    print("Grab failed; retrying...")
                    time.sleep(0.05)
                    continue
                # retrieving frame from buffer
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    print("Retrieve failed; retrying...")
                    time.sleep(0.05)
                    continue

                # flip for mirroring
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    #print("Hand landmarks detected:", results.multi_hand_landmarks is not None) # debug print
                    # Drawing landmarks and connections
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))
                    
                    # normalize landmarks and calculation motion energy
                    curr_lm = normalize_landmarks(hand_landmarks)

                    # Skip if landmarks are empty
                    if curr_lm.size == 0: 
                        continue
                    curr_flat = flatten(curr_lm)

                    if prev_lm is not None:
                        m_energy = motion_energy(curr_lm, prev_lm)
                        #print(f"Motion energy: {m_energy}") # debug print
                        motion_buf.append(m_energy)

                        # smoothed motion
                        smooth_m_energy = np.mean(motion_buf)
                        #print(f"Smoothed motion energy: {smooth_m_energy}")

                        # start detection
                        if not in_segment and smooth_m_energy > START_THRESH:
                            print(f"Start condition met: {smooth_m_energy} > {START_THRESH}")
                            start_counter += 1
                            #print(f"Start counter incrementing: {start_counter}")
                            if start_counter >= DEBOUNCE_START:
                                in_segment = True
                                #print(f"in_segment: {in_segment}")
                                seg_frames = [curr_flat]
                                print(f"Sign started at frame {frame_idx}")
                        else: 
                            start_counter = 0
                        
                        # end detection
                        if in_segment and smooth_m_energy < END_THRESH:
                            print(f"End condition met: {smooth_m_energy} < {END_THRESH}")
                            end_counter += 1
                            print(f"End counter incrementing: {end_counter}")
                            if end_counter >= DEBOUNCE_END:
                                in_segment = False
                                #print(f"in_segment: {in_segment}")
                                print(f"Sign ended at frame {frame_idx}, length: {len(seg_frames)} frames")
                                # TODO: actually pass emebedding to model for classification
                                embedding = np.mean(seg_frames, axis=0) # average embedding
                                print("Embedding:", embedding)
                                num_signs += 1
                                print(f"Total signs captured: {num_signs}")
                                seg_frames = []

                                # run model on embedding
                                output = model(torch.tensor(embedding, dtype=torch.float32))
                                _, predicted = torch.max(output, 0)
                                letter = letter_mapping[int(predicted)]
                                print(f"Predicted letter: {letter}")
                        else:
                            end_counter = 0
                        if in_segment:
                            seg_frames.append(curr_flat)
                            start_counter = 0
                    prev_lm = curr_lm
                    
                # how many signs have been detected
                frame_h, frame_w, _ = frame.shape
                cv2.putText(frame, f"Signs: {num_signs}", (frame_w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

                # predicted letter
                cv2.putText(frame, f"Letter: {letter}", (frame_w - 200, frame_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

                frame_idx += 1
                # calc / rep FPS
                frames += 1
                curr = time.time()
                if curr - prev_time >= 1.0:
                    fps = frames / (curr - prev_time)
                    prev_time = curr
                    frames = 0
                else:
                    fps = None
                if fps:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                # quit key 
                if key == ord('q'):
                    break
                # save key
                if key == ord('s'):
                    fname = f"capture_{saved:03d}.jpg"
                    cv2.imwrite(fname, frame)
                    print("Saved", fname)
                    saved += 1

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
