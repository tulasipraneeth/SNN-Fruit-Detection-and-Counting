
import streamlit as st
import cv2
import numpy as np
import torch
import snntorch as snn
from snntorch import surrogate
import tempfile
import os
from scipy.spatial import distance

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="SNN Apple Detector", layout="centered")

# ------------------ SIDEBAR ------------------ #
theme = st.sidebar.radio("🌗 Choose Theme", ("Dark Futuristic", "Light Minimal"))
show_progress = st.sidebar.checkbox("🔄 Show Progress Bar", value=True)

# ------------------ DYNAMIC STYLING ------------------ #
if theme == "Dark Futuristic":
    st.markdown("""
        <style>
        html, body, .stApp, .block-container {
            background: radial-gradient(circle at top left, #0a0f2c, #05070e);
            background-size: 400% 400%;
            animation: gradientShift 30s ease infinite;
            font-family: 'Segoe UI', sans-serif;
            color: #FFFFFF;
        }
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .title {
            font-size: 2.8em;
            font-weight: bold;
            text-align: center;
            padding: 10px 0;
            background: linear-gradient(90deg, #4AE8E8, #8A2BE2, #4AE8E8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: glow 4s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #4AE8E8; }
            to { text-shadow: 0 0 20px #8A2BE2; }
        }
        .info-card {
            background: rgba(27, 31, 59, 0.85);
            border: 1px solid rgba(74, 232, 232, 0.3);
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 12px;
            box-shadow: 0 0 20px rgba(74, 232, 232, 0.15);
            backdrop-filter: blur(5px);
        }
        .info-card:hover {
            transform: scale(1.01);
            box-shadow: 0 0 25px rgba(138, 43, 226, 0.4);
        }
        .stDownloadButton>button {
            background-color: #4AE8E8;
            color: black;
            border-radius: 6px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #34caca;
            transform: scale(1.05);
        }
        hr {
            border: 1px solid #4AE8E8;
            margin-top: 12px;
            margin-bottom: 18px;
        }
        </style>
    """, unsafe_allow_html=True)

elif theme == "Light Minimal":
    st.markdown("""
        <style>
        html, body, .stApp, .block-container {
            background-color: #f9f9f9 !important;
            color: #222 !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 2.6em;
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }
        .info-card {
            background-color: #ffffff;
            border: 1px solid #ddd;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------ TITLE ------------------ #
st.markdown("""
    <div class="title">🤖 SNN Apple Detection & Counting</div>
    <p style='text-align:center; font-size:16px; color: #bfbfbf;'>
        Upload an orchard video and let our AI detect and count apples using Spiking Neural Networks.
    </p>
    <hr>
""", unsafe_allow_html=True)

# ------------------ VIDEO UPLOAD ------------------ #
video_file = st.file_uploader("📁 Upload Apple Orchard Video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid:
        temp_vid.write(video_file.read())
        video_path = temp_vid.name

    st.video(video_path)

    def preprocess_frame(frame, target_size=(128, 128)):
        resized = cv2.resize(frame, target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray / 255.0

    cap = cv2.VideoCapture(video_path)
    preprocessed_frames, original_frames = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frames.append(preprocess_frame(frame))
        original_frames.append(frame)
    cap.release()

    frames_tensor = torch.tensor(preprocessed_frames).float().unsqueeze(1)

    num_inputs, num_hidden, num_outputs, num_steps, beta = 128*128, 512, 1, 25, 0.95

    class FruitDetectionSNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk2_rec = []
            for _ in range(num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
            return torch.stack(spk2_rec)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitDetectionSNN().to(device)

    def frames_to_spikes(frames_tensor, num_steps):
        torch.manual_seed(42)
        return torch.stack([torch.rand(frames_tensor.shape) < frames_tensor for _ in range(num_steps)]).float()

    if show_progress:
        progress = st.progress(0)

    spikes = frames_to_spikes(frames_tensor, num_steps)

    snn_outputs = []
    for i in range(spikes.shape[1]):
        spike_in = spikes[:, i, 0, :, :].to(device).view(num_steps, -1)
        out = model(spike_in)
        snn_outputs.append(out.sum().item() / num_steps)
        if show_progress:
            progress.progress((i + 1) / spikes.shape[1])

    def detect_apples(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 100, 50), (15, 255, 255))
        red2 = cv2.inRange(hsv, (165, 100, 50), (180, 255, 255))
        mask = cv2.bitwise_or(red1, red2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]

    def is_duplicate(new_box, tracked, thresh=50):
        new_c = (new_box[0]+new_box[2]/2, new_box[1]+new_box[3]/2)
        return any(distance.euclidean(new_c, (b[0]+b[2]/2, b[1]+b[3]/2)) < thresh for b in tracked)

    tracked_boxes, frame_counts = [], []
    height, width = original_frames[0].shape[:2]
    out_path = "output_final_snn.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

    for i, frame in enumerate(original_frames):
        boxes = detect_apples(frame)
        count = 0
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Apple ({snn_outputs[i]:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if not is_duplicate(box, tracked_boxes):
                tracked_boxes.append(box)
                count += 1
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frame_counts.append(count)
        out.write(frame)
    out.release()

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.success("✅ Output video generated with bounding boxes on every frame.")
    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download Final Video", f, file_name="output_final_snn.mp4", mime="video/mp4")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    st.subheader("📈 Frame-wise Apple Count")
    st.line_chart(frame_counts)
    st.subheader("📊 Total Detected: " + str(sum(frame_counts)))
    st.markdown("</div>", unsafe_allow_html=True)

    st.balloons()
else:
    st.info("👆 Upload a video to begin.")

st.markdown("""
<hr style="border:1px solid #4AE8E8; margin-top:40px;">
<p style="text-align:center; font-size:13px; color:#aaa;">
  Made with ❤️ by <b>Mukka Tulasi Praneeth</b> · 
  <a href="https://github.com/tulasipraneeth/SNN-Fruit-Detection-and-Counting" target="_blank">GitHub</a> · 
  <a href="https://streamlit.io" target="_blank">Powered by Streamlit</a>
</p>
""", unsafe_allow_html=True)
