import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

st.set_page_config(layout="wide")
st.title("Real-time Fire Detection APP 🔥")

# Tabs
tab1, tab2, tab3 = st.tabs(["Home", "🗃 Data", "Model"])

# Tab 1: Introduction
tab1.markdown("""
### Welcome to the Real-time Fire Detection Application!
This application uses advanced deep learning and computer vision to detect fire in real-time, providing a crucial tool for enhancing safety in various settings. By leveraging a sophisticated model trained on diverse datasets, the app ensures high accuracy in identifying fire hazards.

### Key Features:
- **Real-time Detection**: Captures live video from a webcam and provides instant alerts for fire detection.
- **Deep Learning Model**: Utilizes a finely-tuned model to accurately detect fire in different environments.
- **User-friendly Interface**: Easy-to-use controls with Streamlit, allowing users to start and stop detection effortlessly.

You can find the source code in the [GitHub Repository🚀](https://github.com/aman9650/Skin_cancer_detection)
""")

# Tab 2: Dataset
tab2.markdown("""
### Dataset 📈:
The model was trained using a dataset of fire and non-fire images sourced from [Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset). This diverse dataset ensures the model can reliably distinguish between fire and non-fire scenarios.

Below are examples of images used in the training dataset:

**Fire Image**:
""")
fire_image = Image.open('fire.103.png')
tab2.image(fire_image, caption='Example Fire Image', width=300)

tab2.markdown("""
**Non-Fire Image**:
""")
non_fire_image = Image.open('non_fire.103.png')
tab2.image(non_fire_image, caption='Example Non-Fire Image', width=300)

# Tab 3: Model
model = load_model('model.h5')
tab3.markdown("""
### Click to start the model 🤖---->
""")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Define VideoProcessor to handle frames and make predictions
class FireDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess the frame for the model
        input_size = (150, 150)
        resized_frame = cv2.resize(img, input_size)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction
        prediction = self.model.predict(input_frame)
        fire_detected = prediction[0][0] > 0.5

        # Display the result
        label = "Fire Detected" if fire_detected else "No Fire"
        color = (0, 0, 255) if fire_detected else (0, 255, 0)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Resize frame for display
        display_frame = cv2.resize(img, (400, 480))

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

# Start the WebRTC stream with reduced canvas size
webrtc_ctx = webrtc_streamer(
    key="fire-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FireDetectionProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
        },
        "audio": False,
    },
    async_processing=True,
)

# Check if the WebRTC context is ready and displaying video
if webrtc_ctx.video_processor:
    tab3.markdown("### Video is running. Check the labels on the video to see detection results.")
else:
    tab3.markdown("### Click on 'Start' to begin video streaming and fire detection.")
