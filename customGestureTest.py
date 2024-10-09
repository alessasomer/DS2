import cv2
import mediapipe as mp
import math

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def recognize_gestures(hand_landmarks):
    # Extract necessary landmarks
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
      
    #Open
    if((index_mcp.y - index_tip.y) > 0.15 and
       (middle_mcp.y - middle_tip.y) > 0.2 and
       (ring_mcp.y - ring_tip.y) > 0.2 and
       (pinky_mcp.y - pinky_tip.y) > 0.15 and
       abs(thumb_mcp.x - thumb_tip.x) > 0.05):
        return "Open_Hand"
    
    #Fist
    if((thumb_tip.x > thumb_mcp.x and pinky_mcp.x > thumb_tip.x) | 
       #special case where thumb is hidden
       (thumb_tip.x < thumb_mcp.x and pinky_tip.x < thumb_tip.x)):
        if (abs(index_mcp.y - index_tip.y) < 0.2 and
            abs(middle_mcp.y - middle_tip.y) < 0.2 and
            abs(ring_mcp.y - ring_tip.y) < 0.2 and
            abs(pinky_mcp.y - pinky_tip.y) < 0.2):
            return "Fist"
    
    #Cowabunga
    if((index_mcp.y - index_tip.y) > 0.15 and
       (middle_mcp.y - middle_tip.y) < 0.1 and
       (ring_mcp.y - ring_tip.y) < 0.1 and
       (pinky_mcp.y - pinky_tip.y) > 0.15 and
       abs(thumb_mcp.x - thumb_tip.x) > 0.05):
        return "Cowabunga"
    
    #Thumb Left
    elif (thumb_tip.x < thumb_mcp.x):
        #remaining four fingers are tucked
        if (index_mcp.y < index_tip.y and
            middle_mcp.y < middle_tip.y and
            ring_mcp.y < ring_tip.y and
            pinky_mcp.y < pinky_tip.y and
            #thumb is extended
            abs(thumb_tip.x - pinky_mcp.x) > 0.15 and
            #index finger is not extended down
            (index_tip.y - index_mcp.y) < 0.2):
            return "Thumb_Left"
    
    #Thumb Right
    elif (thumb_tip.x > thumb_mcp.x and thumb_tip.x > pinky_tip.x):
        #remaining four fingers are tucked
        if (abs(index_mcp.y - index_tip.y) < 0.15 and
            abs(middle_mcp.y - middle_tip.y) < 0.15  and
            abs(ring_mcp.y - ring_tip.y) < 0.15 and
            abs(pinky_mcp.y - pinky_tip.y) < 0.15 and
            #index finger is not extended up
            (index_tip.y - index_mcp.y) < 0.2):
            return "Thumb_Right"
        
    #Point Down
    if index_tip.y > index_mcp.y:
        #remaining three fingers are tucked
        if (abs(middle_mcp.y - middle_tip.y) < 0.2 and
            abs(ring_mcp.y - ring_tip.y) < 0.2 and
            abs (pinky_mcp.y - pinky_tip.y) < 0.2 and
            #pointer is extended
            abs(index_mcp.y - index_tip.y) > 0.2 and
            #thumb is tucked
            (thumb_tip.x - thumb_mcp.x) < 0.15):
            return "Point_Down"
    
    #Point Up
    if index_tip.y < index_mcp.y:
        #remaining three fingers are tucked
        if (middle_mcp.y < middle_tip.y and
            ring_mcp.y < ring_tip.y and
            pinky_mcp.y < pinky_tip.y and
            #pointer is extended
            abs(index_mcp.y - index_tip.y) > 0.15 and
            #thumb is tucked
            (thumb_tip.x - thumb_mcp.x) < 0.15):
            return "Point_Up"
        
    return None

model_path = "/Users/williamiorio/CS376/DS2/gesture_recognizer.task"

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
            
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Flip the image horizontally for a later selfie-view display
            # and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
            # Perform gesture recognition on the image
            results = hands.process(image_rgb)
    
            # Draw the gesture recognition results on the image
            if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
                        # Recognize gesture
                        # gesture = recognize_palm(hand_landmarks)
                        gesture = recognize_gestures(hand_landmarks)
                        
                        # Display gesture near hand location
                        cv2.putText(image, gesture,
                                    (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                     int(hand_landmarks.landmark[0].y * image.shape[0]) - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    
            # Display the resulting image
            cv2.imshow('Gesture Recognition', image)
    
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
