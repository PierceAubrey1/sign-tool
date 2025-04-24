
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import time

# Configuration
SIGNS = ["hello", "thank_you", "yes"]
SIGN_IMAGES = {
    "hello": "signs/hello.png",
    "thank_you": "signs/thank_you.png",
    "yes": "signs/yes.png"
}

# Session state init
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "result_text" not in st.session_state:
    st.session_state.result_text = ""
if "detected" not in st.session_state:
    st.session_state.detected = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "started" not in st.session_state:
    st.session_state.started = False
if "hold_start" not in st.session_state:
    st.session_state.hold_start = None

# Style
st.markdown(
    """
    <style>
        body {
            background-color: #143256;
            color: white;
        }
        .main {
            background-color: #143256;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #143256;
            color: white;
            border-radius: 8px;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True
)

# Start screen
if not st.session_state.started:
    st.title("🤟 Welcome to the Sign Language Trainer")
    st.markdown("Learn and practise recognising basic signs using your webcam.")
    if st.button("Start Training"):
        st.session_state.started = True
        st.session_state.result_text = ""
        st.session_state.detected = False
        st.session_state.hold_start = None
    st.stop()

# Main layout
st.title("🤟 Sign Language Trainer")
st.markdown(f"### Sign {st.session_state.current_index + 1} of {len(SIGNS)}")

current_sign = SIGNS[st.session_state.current_index]
image_path = SIGN_IMAGES[current_sign]

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    st.image(image_path, caption=f"This means: {current_sign.replace('_', ' ').title()}", width=300)
    st.markdown(f"### Try this sign: **{current_sign.replace('_', ' ').title()}**")
    if st.session_state.result_text:
        st.success(st.session_state.result_text)
    else:
        st.info("Waiting for correct sign...")

with col2:
    FRAME_WINDOW = st.image([])
    hold_status = st.empty()

# Load references
def load_references(sign):
    with open(f"reference_{sign}_left.json", "r") as f:
        left = json.load(f)
    with open(f"reference_{sign}_right.json", "r") as f:
        right = json.load(f)
    return left, right

reference_left, reference_right = load_references(current_sign)

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def compare_landmarks(current, reference):
    if len(current) != len(reference):
        return False
    total_distance = sum(
        np.sqrt((r['x'] - c['x'])**2 + (r['y'] - c['y'])**2 + (r['z'] - c['z'])**2)
        for r, c in zip(reference, current)
    )
    avg_distance = total_distance / len(reference)
    return avg_distance < 0.15

# Detection loop (supports both hands)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected = False
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                reference = reference_left if hand_label == "Left" else reference_right
                current_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
                if compare_landmarks(current_landmarks, reference):
                    detected = True
                    break

        if detected:
            if st.session_state.hold_start is None:
                st.session_state.hold_start = time.time()
            held_time = time.time() - st.session_state.hold_start
            hold_status.success(f"Perfect, hold for just a moment... ({held_time:.1f}s)")
            if held_time >= 2:
                st.session_state.result_text = "✅ Well done! That's correct."
                st.session_state.detected = True
                st.session_state.hold_start = None
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(1)
                break
        else:
            st.session_state.hold_start = None
            hold_status.warning("Say ahhh... try again and hold for at least 2 seconds")

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# Proceed to next sign
if st.session_state.detected:
    col2.success("Well done, proceed to the next sign!")
    if col2.button("➡️ Next Sign"):
        st.session_state.score += 1
        st.session_state.current_index += 1
        if st.session_state.current_index >= len(SIGNS):
            st.success("🎉 You've completed all the signs!")
            st.session_state.current_index = 0
            st.session_state.score = 0
        st.session_state.result_text = ""
        st.session_state.detected = False
        st.experimental_rerun()

# Show reset button if all signs completed
if st.session_state.score == len(SIGNS):
    if st.button("🔁 Restart Training"):
        st.session_state.current_index = 0
        st.session_state.score = 0
        st.session_state.result_text = ""
        st.session_state.detected = False
        st.session_state.started = False
        st.experimental_rerun()
