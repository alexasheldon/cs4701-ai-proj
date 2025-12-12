import time
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

CAM_INDEX = 0
WINDOW_NAME = "MediaPipe Hands - press q to quit, s to save"

# default threshold for detecting gesture start / end
START_THRESH = 0.015 # originally 0.015
END_THRESH = 0.010 # originally 0.010 (must be less than START THRESH)
DEBOUNCE_START = 3
DEBOUNCE_END = 3 # originally 8
SMOOTH_WINDOW = 7 # originally 5
MIN_SEG_FRAMES = 6

CAL_INTERVAL = 5  # seconds between calibration prompts
MOTION_HISTORY_MAX = 3000 # max length of motion history deque

CNN_MODEL_NAME = "models/cnn_model_3.pt"
MODEL_NAME = "models/mlp_model_norm.pt"
CONFIDENCE_THRESH = 0.0

model_in_h = 200
model_in_w = 200

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

    cnn_letter = "?"
    cnn_percent = 0.0
    mlp_letter = "?"
    mlp_percent = 0.0

    # Load model
    cnn_model = torch.load(CNN_MODEL_NAME)
    model = torch.load(MODEL_NAME, weights_only=False)

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
                    # for hand_landmarks in results.multi_hand_landmarks:
                    #     mp_drawing.draw_landmarks(
                    #         frame,
                    #         hand_landmarks,
                    #         mp_hands.HAND_CONNECTIONS,
                    #         mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                    #         mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    #         )
                    h, w, c = frame.shape
                    hand_landmarks = results.multi_hand_landmarks[-1]
            
                    # Collect pixel coordinates for bounding box
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    
                    # Bounding box
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # # Crop the hand region to send to CNN
                    # cropped_hand = frame[y_min:y_max, x_min:x_max]
                    # cv2.imshow("Cropped Hand", cropped_hand) # show cropped hand
                    # cropped_resized = cv2.resize(cropped_hand, (model_in_h, model_in_w)) # resize to model input size
                    # cropped_tensor = torch.from_numpy(cropped_resized).float() / 255.0 # normalize to [0,1]
                    # cropped_tensor = (cropped_tensor - 0.5) / 0.5 # normallize to [-1,1]
                    # cropped_tensor_reorder = cropped_tensor.permute(2, 0, 1) # change to (Channels, Height, Width) from (H,W,C)
                    # cropped_input = cropped_tensor_reorder.unsqueeze(0) # adding batch dimension (though this can be done with tensor too)
                    # cnn_output = cnn_model(cropped_input) # run through CNN model
                    # _, cnn_predicted = torch.max(cnn_output, 1)
                    # letter = letter_mapping[int(cnn_predicted)]

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

                                # Crop the hand region to send to CNN
                                cropped_hand = frame[y_min:y_max, x_min:x_max]
                                cv2.imshow("Cropped Hand", cropped_hand) # show cropped hand
                                cropped_resized = cv2.resize(cropped_hand, (model_in_h, model_in_w)) # resize to model input size
                                cropped_tensor = torch.from_numpy(cropped_resized).float() / 255.0 # normalize to [0,1]
                                cropped_tensor = (cropped_tensor - 0.5) / 0.5 # normallize to [-1,1]
                                cropped_tensor_reorder = cropped_tensor.permute(2, 0, 1) # change to (Channels, Height, Width) from (H,W,C)
                                cropped_input = cropped_tensor_reorder.unsqueeze(0) # adding batch dimension (though this can be done with tensor too)
                                cnn_output = cnn_model(cropped_input) # run through CNN model
                                _, cnn_predicted = torch.max(cnn_output, 1)
                                cnn_letter = letter_mapping[int(cnn_predicted)]
                                cnn_percent = torch.max(torch.softmax(cnn_output, dim=1), 1).values

                                num_signs += 1
                                print(f"Total signs captured: {num_signs}")
                                seg_frames = []

                                # run model on embedding
                                output = model(torch.tensor(embedding, dtype=torch.float32))
                                _, predicted = torch.max(output, 0)
                                mlp_letter = letter_mapping[int(predicted)]
                                mlp_percent = torch.max(torch.softmax(output, dim=0), 0).values
                                print(f"Predicted letter: {letter}")
                                print(f"Confidence: {percent}")
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
                if cnn_percent > mlp_percent:
                    percent = cnn_percent
                    letter = cnn_letter
                    model_type = "CNN"
                else:
                    percent = mlp_percent
                    letter = mlp_letter
                    model_type = "MLP"

                if percent >= CONFIDENCE_THRESH:
                    letter_text = f"Letter: {letter}, {round(float(percent),2)}, {model_type}"
                else:
                    letter_text = "Letter: ?"
                cv2.putText(frame, letter_text, (frame_w - 500, frame_h - 30),
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
