#2.15.0
#2.16.1
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(layout="wide")
st.title("Real-time Fire Detection APP 🔥")
## -------------------------------------------------------
# Tab 1
tab1, tab2, tab3= st.tabs(["Home", "🗃 Data", "Model"])

# Display content in Tab 1
tab1.markdown("""
###
Welcome to the Real-time Fire Detection Application!
This application uses advanced deep learning and computer vision to detect fire in real-time, providing a crucial tool for enhancing safety in various settings. By leveraging a sophisticated model trained on diverse datasets, the app ensures high accuracy in identifying fire hazards.

### Key Features:
- **Real-time Detection**: Captures live video from a webcam and provides instant alerts for fire detection.
- **Deep Learning Model**: Utilizes a finely-tuned model to accurately detect fire in different environments.
- **User-friendly Interface**: Easy-to-use controls with Streamlit, allowing users to start and stop detection effortlessly.

You can find the source code in the [GitHub Repository🚀](https://github.com/aman9650/Skin_cancer_detection)
""")



### ------------------------------------------------------------
# Tab 2

# Display content in Tab 2
tab2.markdown("""
### Dataset 📈:
The model was trained using a dataset of fire and non-fire images sourced from [Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset). This diverse dataset ensures the model can reliably distinguish between fire and non-fire scenarios.

Below are examples of images used in the training dataset:

**Fire Image**:
""")

# Display an example fire image
fire_image = Image.open('fire.103.png')
tab2.image(fire_image, caption='Example Fire Image', width=500)

tab2.markdown("""
**Non-Fire Image**:
""")

# Display an example non-fire image
non_fire_image = Image.open('non_fire.103.png')
tab2.image(non_fire_image, caption='Example Non-Fire Image', width=500)




## ------------------------------------------------------

# Tab3


# Load your trained model
model = load_model('model.h5')

# Set the title of the app
tab3.markdown("""
### Click to start the model 🤖---->
""")


# Placeholder for video frames
frame_placeholder = tab3.empty()

# Function to capture video
def capture_video(stop_signal):
    cap = cv2.VideoCapture(1)  # 0 is the default camera

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    while not st.session_state[stop_signal]:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Preprocess the frame for your model
        input_size = (150, 150)
        resized_frame = cv2.resize(frame, input_size)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(input_frame)
        fire_detected = prediction[0][0] > 0.5  # Assuming binary classification with a threshold

        # Display the result
        if fire_detected:
            label = "Fire Detected"
            color = (0, 0, 255)  # Red
        else:
            label = "No Fire"
            color = (0, 255, 0)  # Green

        # Put label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the resulting frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

    # Release the webcam
    cap.release()
    frame_placeholder.empty()  # Clear the frame placeholder when done

# Initialize session state
if 'stop_signal' not in st.session_state:
    st.session_state.stop_signal = False

# Start button
if tab3.button("Start Fire Detection"):
    st.session_state.stop_signal = False
    capture_video('stop_signal')

# Stop button
if tab3.button("Stop Fire Detection"):
    st.session_state.stop_signal = True

