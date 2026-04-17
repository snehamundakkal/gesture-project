import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
prev_gesture = None
last_action_time = time.time()

def detect_swipe(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    diff = index_tip.x - wrist.x
    if diff > 0.2:
        return "right"
    elif diff < -0.2:
        return "left"
    return None

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_swipe(hand_landmarks)

            if gesture and gesture != prev_gesture and time.time() - last_action_time>1.5:
                if gesture == "right":
                    pyautogui.hotkey('alt', 'tab')
                    print("Switched to next window")
                elif gesture == "left":
                    pyautogui.hotkey('alt', 'tab')
                    print("Switched to previous window")
                prev_gesture = gesture
                last_action_time = time.time()
    else:
        prev_gesture = None

    cv2.imshow("Gesture Window Switcher", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()