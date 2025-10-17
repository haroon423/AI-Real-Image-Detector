import streamlit as st
from transformers import pipeline, VideoMAEImageProcessor, VideoMAEForVideoClassification
from PIL import Image
import torch, cv2, numpy as np, tempfile, os, warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Generated Media Detection Suite", layout="centered")

st.title("🤖 Unified AI-Generated Media Detection Suite")
st.write("Detect AI-generated or deepfake **images and videos** using multiple advanced models.")

# Sidebar model selection
model_choice = st.sidebar.radio(
    "🧩 Select Detection Model",
    [
        "AIRealNet (AI vs Real Image)",
        "ViT Deepfake Detection",
        "Deepfake-v1 Model",
        "VideoMAE (AI vs Real Video)"
    ]
)

# --- 1️⃣ AIRealNet Model ---
if model_choice == "AIRealNet (AI vs Real Image)":
    @st.cache_resource
    def load_airealnet():
        return pipeline("image-classification", model="XenArcAI/AIRealNet")

    pipe = load_airealnet()
    uploaded = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Run AIRealNet Detection"):
            with st.spinner("Running AIRealNet model..."):
                preds = pipe(img)
            result = {p["label"].lower(): p["score"] for p in preds}
            st.subheader("📊 Model Output")
            st.json(result)

            ai_score = result.get("artificial", 0)
            real_score = result.get("real", 0)

            if ai_score > real_score:
                st.markdown("### 🏆 Final Decision: 🔴 AI-Generated")
                st.markdown(f"🎯 Confidence: **{ai_score:.2f}**")
            else:
                st.markdown("### 🏆 Final Decision: 🟢 Real")
                st.markdown(f"🎯 Confidence: **{real_score:.2f}**")

# --- 2️⃣ ViT Deepfake Model ---
elif model_choice == "ViT Deepfake Detection":
    @st.cache_resource
    def load_deepfake_vit():
        return pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")

    deepfake_model = load_deepfake_vit()
    uploaded = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Run ViT Deepfake Detection"):
            st.info("Running ViT deepfake detection model... ⏳")
            preds = deepfake_model(img)
            pred_dict = {p['label'].lower(): p['score'] for p in preds}

            real_score = pred_dict.get("real", 0)
            fake_score = pred_dict.get("fake", 0)

            if real_score > fake_score:
                final_label = "🟢 Real"
                confidence = real_score
            else:
                final_label = "🔴 AI-Generated"
                confidence = fake_score

            st.subheader("📊 Model Outputs")
            st.json(pred_dict)
            st.markdown(f"### 🏆 Final Decision: {final_label}")
            st.markdown(f"🎯 Confidence: **{confidence:.2f}**")

# --- 3️⃣ Deepfake-v1 (PrithivMLmods) Model ---
elif model_choice == "Deepfake-v1 Model":
    @st.cache_resource
    def load_deepfake_v1():
        return pipeline("image-classification", model="PrithivMLmods/deepfake-detector-model-v1")

    dfv1_model = load_deepfake_v1()
    uploaded = st.file_uploader("🧬 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Deepfake-v1 Detection"):
            st.info("Running deepfake-v1 model... ⏳")
            preds = dfv1_model(img)
            pred_dict = {p['label'].lower(): p['score'] for p in preds}

            real_score = pred_dict.get("real", 0)
            fake_score = pred_dict.get("fake", 0)

            if real_score > fake_score:
                final_label = "🟢 Real"
                confidence = real_score
            else:
                final_label = "🔴 AI-Generated"
                confidence = fake_score

            st.subheader("📊 Model Output")
            st.json(pred_dict)
            st.markdown(f"### 🏆 Final Decision: {final_label}")
            st.markdown(f"🎯 Confidence: **{confidence:.2f}**")

# --- 4️⃣ VideoMAE Model ---
elif model_choice == "VideoMAE (AI vs Real Video)":
    MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

    @st.cache_resource
    def load_videomae():
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return processor, model, device

    def extract_frames(video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // num_frames)
        frames = []
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        return frames[:num_frames]

    def run_inference(processor, model, device, frames):
        if not frames or len(frames) < 16:
            st.error("Not enough frames extracted.")
            return None, None, None
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        max_logit = torch.max(logits, dim=1).values.item()
        threshold = 5.0
        is_ai = max_logit < threshold
        confidence = 1.0 / (1.0 + np.exp(-abs(max_logit - threshold)))
        label = "AI-Generated" if is_ai else "Real"
        scores = [confidence, 1 - confidence] if is_ai else [1 - confidence, confidence]
        return label, confidence, scores

    if 'videomae_loaded' not in st.session_state:
        with st.spinner("Loading VideoMAE model (~2GB, first time only)..."):
            processor, model, device = load_videomae()
            st.session_state.videomae_loaded = True
            st.session_state.processor = processor
            st.session_state.model = model
            st.session_state.device = device
        st.success("VideoMAE model loaded!")
    else:
        processor = st.session_state.processor
        model = st.session_state.model
        device = st.session_state.device

    uploaded_video = st.file_uploader("🎬 Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_video:
        with st.spinner("Processing video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=tempfile.gettempdir()) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            frames = extract_frames(tmp_path)
            label, confidence, scores = run_inference(processor, model, device, frames)

            try:
                os.unlink(tmp_path)
            except:
                pass

            if label:
                st.success(f"**Prediction**: {label} (Confidence: {confidence:.2f})")
                st.write(f"AI-Generated: {scores[0]:.2f} | Real: {scores[1]:.2f}")
                st.video(uploaded_video)

# --- Footer ---
st.markdown("---")
st.caption("🧠 Powered by Hugging Face Transformers | Built by Haroon Khan")
