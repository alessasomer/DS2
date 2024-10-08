import cv2
import mediapipe as mp

# Import the tasks API for gesture recognition
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

import webbrowser
import time
import pyautogui

# Path to the gesture recognition model
#model_path = "/Users/williamiorio/CS376/DS2/gesture_recognizer.task"
model_path = "gesture_recognizer.task"

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    run = 0
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
        result = gesture_recognizer.recognize(mp_image)

        # Draw the gesture recognition results on the image
        if result.gestures:
            recognized_gesture = result.gestures[0][0].category_name
            confidence = result.gestures[0][0].score

            # Example of taking browser action based on recognized gesture
            if recognized_gesture == "Open_Palm":
                    if(run==0):
                        webbrowser.open('https://poki.com/en/g/subway-surfers', new=2)
                        time.sleep(2)
                        pyautogui.click(1025,674)
                        pyautogui.moveTo(1512,982)
                        run = 1
                 
            if recognized_gesture == "Thumb_Up":
                 pyautogui.press('up')
                 time.sleep(0.5)
            
            if recognized_gesture == "Thumb_Down":
                 pyautogui.press('down')
                 time.sleep(0.5) 

            if recognized_gesture == "Pointing_Up":
                 pyautogui.press('left')
                 time.sleep(0.5)

            if recognized_gesture == "Victory":
                 pyautogui.press('right')
                 time.sleep(0.5)

            if recognized_gesture == "ILoveYou":
                 pyautogui.press('space')
                 time.sleep(0.5)


                    

            # Display recognized gesture and confidence
            cv2.putText(image, f"Gesture: {recognized_gesture} ({confidence:.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
