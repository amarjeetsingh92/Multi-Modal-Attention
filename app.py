# integrated_app.py
# Single-window Classroom Intelligence Suite
#   - Live webcam / MP4: face gaze + face emotion + attention scoring
#   - Live mic: rolling 10-second chunks → Whisper-tiny transcription → text emotion
#   - Speech emotion overlaid as a coloured banner on the video frame
#   - Unified dashboard: attention chart + speech emotion timeline (live)
#   - Lecture RAG: upload lecture text → ask questions via FAISS + LLM

# =====================================================================
# IMPORTS
# =====================================================================
import streamlit as st
import altair as alt

import os, csv, time, io, threading, queue, tempfile
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import cv2
from mediapipe.python.solutions import face_mesh as _mp_face_mesh_mod
from mediapipe.python.solutions import drawing_utils as _mp_drawing_mod
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import timm
from transformers import (
    pipeline,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
import sounddevice as sd
import soundfile as sf

# RAG imports
import re
from pathlib import Path
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Classroom Intelligence Suite",
    layout="wide",
    page_icon="🎓",
)

# =====================================================================
# CONSTANTS
# =====================================================================
MIC_SAMPLE_RATE  = 16000          # Hz — Whisper's native rate
MIC_CHUNK_SEC    = 10             # seconds per transcription chunk
CSV_PATH         = "attention_log.csv"
TRANSCRIPT_PATH  = "transcript_log.txt"

RAG_CHUNK_SIZE    = 400
RAG_CHUNK_OVERLAP = 80
RAG_TOP_K         = 5

EMOTION_COLORS = {                # BGR for OpenCV overlay
    "joy":      (  0, 220, 100),
    "anger":    (  0,   0, 230),
    "sadness":  (200, 100,   0),
    "fear":     (150,   0, 200),
    "disgust":  (  0, 150, 150),
    "surprise": (  0, 180, 255),
    "neutral":  (180, 180, 180),
    "silence":  (100, 100, 100),
}

# =====================================================================
# MODEL LOADERS
# =====================================================================

@st.cache_resource
def load_whisper():
    """Load Whisper-tiny on-device — no API key needed."""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model     = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-tiny").to(device)
    model.eval()
    return processor, model, device


@st.cache_resource
def load_hf_text_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )


@st.cache_resource
def load_hf_image_emotion_model(model_name: str):
    try:
        return pipeline(
            "image-classification", model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception as e:
        st.warning(f"[Face emotion model] load failed: {e}")
        return None


class ViTGazeRegressor(nn.Module):
    def __init__(self, vit_name="vit_base_patch16_224", pretrained=True,
                 hidden_dim=256, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            vit_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


@st.cache_resource
def load_gaze_model(vit_name: str, checkpoint_bytes=None):
    """Robust checkpoint loading — handles plain state_dict, wrapped, or whole-model saves."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ViTGazeRegressor(vit_name=vit_name, pretrained=True).to(device)
    if checkpoint_bytes is not None:
        try:
            state = torch.load(checkpoint_bytes, map_location=device)
            if isinstance(state, dict) and any(
                k.startswith("head") or k.startswith("backbone") for k in state.keys()
            ):
                model.load_state_dict(state, strict=False)
            elif isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            st.success("Gaze checkpoint loaded.")
        except Exception as e:
            st.warning(f"Gaze checkpoint error: {e}")
    model.eval()
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, device, tfm


# =====================================================================
# RAG HELPERS
# =====================================================================

@st.cache_resource
def load_embedding_model():
    if not ST_AVAILABLE:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"[RAG] Embedding model load failed: {e}")
        return None


def _chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE,
                overlap: int = RAG_CHUNK_OVERLAP):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in (". ", "? ", "! ", "\n\n", "\n"):
                pos = text.rfind(sep, start + overlap, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def _build_rag_index(text: str, embed_model):
    if not FAISS_AVAILABLE:
        return None, None
    chunks = _chunk_text(text)
    if not chunks:
        return [], None
    embeddings = embed_model.encode(chunks, show_progress_bar=False,
                                    convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return chunks, index


def _retrieve_chunks(query, chunks, index, embed_model, top_k=RAG_TOP_K):
    if index is None or not chunks:
        return []
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(q_vec, min(top_k, len(chunks)))
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def _rag_answer(query, context_chunks, api_key="",
                model="meta/llama-3.1-70b-instruct"):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful study assistant. Answer the student's question using ONLY the "
        "lecture material provided below. If the answer is not in the material, say so clearly.\n\n"
        f"=== LECTURE MATERIAL ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n=== ANSWER ==="
    )
    if api_key and OPENAI_AVAILABLE:
        try:
            client = openai.OpenAI(api_key=api_key,
                                   base_url="https://integrate.api.nvidia.com/v1")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600, temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ LLM API error: {e}\n\n**Retrieved context:**\n\n{context}"
    return "*(No LLM API key — showing raw retrieved passages)*\n\n" + context


# =====================================================================
# VISION HELPERS  (3-layer gaze cascade from x.py)
# =====================================================================

def _avg_pt(lm, idx, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in idx if i < len(lm)]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _gaze_from_angles(p, y, yt=15., pt=12.):
    if y < -yt: return "right"
    if y >  yt: return "left"
    if p < -pt: return "up"
    if p >  pt: return "down"
    return "front"


def _iris_gaze(lm, w, h):
    """Iris-position gaze with wider thresholds (0.38/0.62) to catch side glances."""
    try:
        li = _avg_pt(lm, [468, 469, 470, 471], w, h)
        ri = _avg_pt(lm, [473, 474, 475, 476], w, h)
        if not li or not ri:
            return "front"
        lw = (lm[133].x * w - lm[33].x * w)  or 1e-6
        rw = (lm[263].x * w - lm[362].x * w) or 1e-6
        ir = ((li[0] - lm[33].x * w) / lw +
              (ri[0] - lm[362].x * w) / rw) / 2
        lt = _avg_pt(lm, [159, 160], w, h); lb = _avg_pt(lm, [145, 144], w, h)
        rt = _avg_pt(lm, [386, 387], w, h); rb = _avg_pt(lm, [374, 380], w, h)
        vr = 0.5 if None in (lt, lb, rt, rb) else (
            ((li[1] - lt[1]) / ((lb[1] - lt[1]) or 1e-6) +
             (ri[1] - rt[1]) / ((rb[1] - rt[1]) or 1e-6)) / 2)
        if ir < 0.38: return "right"
        if ir > 0.62: return "left"
        if vr < 0.38: return "up"
        if vr > 0.70: return "down"
        return "front"
    except Exception:
        return "front"


# 3-D canonical face model for solvePnP head-pose gaze
_FACE_3D = np.array([
    [ 0.0,   0.0,   0.0 ],   # nose tip   (4)
    [ 0.0, -63.6, -12.5 ],   # chin       (152)
    [-43.3,  32.7, -26.0],   # left outer (33)
    [ 43.3,  32.7, -26.0],   # right outer(263)
    [-28.9, -28.9, -24.1],   # left mouth (61)
    [ 28.9, -28.9, -24.1],   # right mouth(291)
], dtype=np.float64)
_FACE_IDX = [4, 152, 33, 263, 61, 291]


def _head_pose_gaze(lm, w, h, yaw_thr=18., pitch_thr=15.):
    """Layer-1 gaze: pure geometry via solvePnP — no checkpoint required."""
    try:
        pts2d  = np.array([[lm[i].x * w, lm[i].y * h] for i in _FACE_IDX],
                           dtype=np.float64)
        cam_mx = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D, pts2d, cam_mx, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        yaw_deg   = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        pitch_deg = float(np.degrees(np.arcsin(-rmat[2, 0])))
        if   yaw_deg   < -yaw_thr:   return "right"
        elif yaw_deg   >  yaw_thr:   return "left"
        elif pitch_deg < -pitch_thr: return "up"
        elif pitch_deg >  pitch_thr: return "down"
        else:                        return "front"
    except Exception:
        return None


def normalize_label(raw_label: str) -> str:
    if not raw_label:
        return "unknown"
    mapping = {
        "happiness": "happy",    "happy":     "happy",
        "surprise":  "surprise", "surprised": "surprise",
        "sadness":   "sadness",  "sad":       "sadness",
        "neutral":   "neutral",  "neutrality":"neutral",
        "angry":     "angry",    "anger":     "angry",
        "disgust":   "disgust",  "fear":      "fear",
    }
    token = "".join(c for c in raw_label.lower() if c.isalpha() or c.isspace()).split()
    base  = token[-1] if token else raw_label.lower()
    return mapping.get(base, base or "unknown")


def predict_emotion_batch(pipe, faces_rgb: list) -> list:
    if pipe is None or not faces_rgb:
        return ["unknown"] * len(faces_rgb)
    try:
        imgs    = [Image.fromarray(f.astype("uint8"), "RGB") for f in faces_rgb]
        results = pipe(imgs, top_k=1)
        if isinstance(results[0], list):
            return [normalize_label(r[0].get("label", "") if r else "") for r in results]
        return [normalize_label(results[0].get("label", ""))] * len(faces_rgb)
    except Exception:
        out = []
        for f in faces_rgb:
            try:
                r = pipe(Image.fromarray(f.astype("uint8"), "RGB"), top_k=1)
                out.append(normalize_label(r[0].get("label", "") if r else ""))
            except Exception:
                out.append("unknown")
        return out


def predict_gaze_angles_batch(model, device, transform, faces_rgb_list):
    if not faces_rgb_list:
        return []
    tensors = [transform(Image.fromarray(f.astype("uint8"), "RGB"))
               for f in faces_rgb_list]
    x = torch.stack(tensors).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy()
    return [(float(o[0]), float(o[1])) for o in out]


def _append_csv(ts, elapsed, attn, cls, subj):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "elapsed_seconds", "attention_level",
                        "class_name", "subject_name"])
        w.writerow([ts, elapsed, attn, cls, subj])


def _save_transcript_chunk(text: str, cls: str, subj: str):
    """Persist each transcription chunk immediately to disk."""
    if not text:
        return
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] ({cls} / {subj})  {text}\n")


# =====================================================================
# BACKGROUND INFERENCE WORKER  (x.py FIX 1 — prevents frame freeze)
# =====================================================================

def _inference_worker(stop_evt: threading.Event,
                      in_q: queue.Queue, out_q: queue.Queue):
    """
    Consumes face-crop batches, runs gaze + emotion inference, posts results.
    Runs in a daemon thread so the render loop never blocks on inference.
    """
    while not stop_evt.is_set():
        try:
            job = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        (faces_rgb_list, bbox_list, lm_list,
         gaze_model, gaze_device, gaze_tfm,
         face_emo_pipe, yaw_thr, pitch_thr, w, h) = job

        try:
            gaze_angles = (
                predict_gaze_angles_batch(gaze_model, gaze_device, gaze_tfm, faces_rgb_list)
                if faces_rgb_list and gaze_model is not None
                else [(0., 0.)] * len(faces_rgb_list)
            )
        except Exception:
            gaze_angles = [(0., 0.)] * len(faces_rgb_list)

        emo_labels = predict_emotion_batch(face_emo_pipe, faces_rgb_list)

        faces_info = []
        for idx in range(len(faces_rgb_list)):
            emo    = emo_labels[idx]
            lm_obj = lm_list[idx]

            # Layer 1: solvePnP head pose (geometry, always available)
            gaze_lbl = _head_pose_gaze(lm_obj.landmark, w, h, yaw_thr, pitch_thr) \
                       if lm_obj is not None else None

            # Layer 2: iris position (if head pose returned None)
            if gaze_lbl is None and lm_obj is not None:
                gaze_lbl = _iris_gaze(lm_obj.landmark, w, h)

            # Layer 3: ViT angle model (only overrides "front" if model is confident)
            if gaze_lbl in (None, "front") and gaze_angles:
                p_deg, y_deg = gaze_angles[idx]
                angle_lbl = _gaze_from_angles(p_deg, y_deg, yaw_thr, pitch_thr)
                if angle_lbl != "front":
                    gaze_lbl = angle_lbl

            gaze_lbl = gaze_lbl or "front"

            faces_info.append({
                "bbox":          bbox_list[idx],
                "emotion_label": emo,
                "gaze":          gaze_lbl,
            })

        out_q.put(faces_info)


# =====================================================================
# AUDIO PIPELINE — live mic + Whisper + emotion  (from x.py)
# =====================================================================

def _audio_worker(stop_evt: threading.Event,
                  whisper, text_model,
                  result_q: queue.Queue, error_q: queue.Queue,
                  mic_device_idx=None):
    """
    Continuously records MIC_CHUNK_SEC seconds of mic audio,
    transcribes with on-device Whisper-tiny, classifies emotion,
    and posts results to result_q for the main thread to consume.
    Runs in a daemon thread; stops when stop_evt is set.
    """
    chunk_samps = int(MIC_SAMPLE_RATE * MIC_CHUNK_SEC)
    elapsed_ref = time.time()

    while not stop_evt.is_set():
        # ── record one chunk ─────────────────────────────────────────
        try:
            rec_kwargs = dict(samplerate=MIC_SAMPLE_RATE,
                              channels=1, dtype="float32")
            if mic_device_idx is not None:
                rec_kwargs["device"] = mic_device_idx
            audio = sd.rec(chunk_samps, **rec_kwargs)
            sd.wait()
        except Exception as e:
            error_q.put(str(e))
            break

        if stop_evt.is_set():
            break

        audio_1d = audio.flatten()

        # ── transcribe with Whisper-tiny ─────────────────────────────
        try:
            processor, wmodel, wdevice = whisper
            inputs = processor(
                audio_1d, sampling_rate=MIC_SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(wdevice)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe")
            with torch.no_grad():
                predicted_ids = wmodel.generate(
                    inputs, forced_decoder_ids=forced_decoder_ids)
            text = processor.batch_decode(
                predicted_ids, skip_special_tokens=True)[0].strip()
        except Exception as te:
            error_q.put(f"Whisper error: {te}")
            text = ""

        # ── classify emotion ─────────────────────────────────────────
        if text:
            try:
                res   = text_model(text)[0]
                top   = max(res, key=lambda x: x["score"])
                emo   = top["label"]
                score = float(top["score"])
            except Exception:
                emo, score = "neutral", 0.0
        else:
            emo, score = "silence", 0.0

        t_offset = int(time.time() - elapsed_ref)
        result_q.put({"time": t_offset, "emotion": emo,
                      "score": score,   "text": text})


# =====================================================================
# SESSION STATE DEFAULTS
# =====================================================================
for k, v in {
    "latest_speech_emotion": None,
    "latest_speech_score":   0.0,
    "audio_timeline":        [],
    "full_transcript":       "",
    "audio_error":           None,
    "audio_stop_evt":        None,   # threading.Event when mic is running
    "audio_result_q":        None,   # results queue: worker → main thread
    "audio_error_q":         None,   # errors queue:  worker → main thread
    "whisper_pipe":          None,
    "hf_text_pipeline":      None,
    "hf_face_pipeline":      None,
    "gaze_bundle":           None,
    "infer_stop_evt":        None,   # background inference thread
    "infer_in_q":            None,
    "infer_out_q":           None,
    "rag_chunks":            None,
    "rag_index":             None,
    "rag_lecture_name":      None,
    "rag_chat_history":      [],
    "rag_embed_model":       None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================================
# SIDEBAR — settings
# =====================================================================
st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("🏫 Session")
class_name    = st.sidebar.text_input("Class Name",  "Enter Class Name")
subject_name  = st.sidebar.text_input("Subject",     "Enter Subject Name")
start_monitor = st.sidebar.checkbox("▶ Start Monitoring", value=False)

st.sidebar.subheader("📹 Video Source")
video_source = st.sidebar.radio("Source", ["Webcam", "MP4 file"], index=0)
uploaded_mp4 = None
if video_source == "MP4 file":
    uploaded_mp4 = st.sidebar.file_uploader(
        "Upload MP4", type=["mp4", "avi", "mov"])
    if uploaded_mp4 is None:
        st.sidebar.info("Upload a video file to begin.")

st.sidebar.subheader("🎯 Gaze Model")
vit_model_name  = st.sidebar.selectbox(
    "Backbone", ["vit_base_patch16_224", "vit_small_patch16_224"], index=0)
gaze_checkpoint = st.sidebar.file_uploader("Gaze checkpoint (.pt)", type=["pt"])
yaw_threshold   = st.sidebar.slider("Yaw threshold (°)",   5.0, 30.0, 15.0)
pitch_threshold = st.sidebar.slider("Pitch threshold (°)", 5.0, 30.0, 12.0)

st.sidebar.subheader("😐 Face Emotion Model")
hf_face_name      = st.sidebar.text_input("HF model", "trpakov/vit-face-expression")
reload_face_model = st.sidebar.button("(Re)load face emotion model")
max_faces         = st.sidebar.slider("Max faces", 1, 6, 4)
log_every         = st.sidebar.number_input("CSV log interval (s)", 30, 600, 60)

st.sidebar.subheader("🎙️ Microphone")
mic_devices = []
try:
    devs = sd.query_devices()
    mic_devices = [f"{i}: {d['name']}"
                   for i, d in enumerate(devs)
                   if d["max_input_channels"] > 0]
except Exception:
    pass

selected_mic = st.sidebar.selectbox(
    "Input device", mic_devices if mic_devices else ["Default"], index=0)
mic_device_idx = None
if mic_devices and selected_mic != "Default":
    try:
        mic_device_idx = int(selected_mic.split(":")[0])
        sd.default.device = mic_device_idx, None
    except Exception:
        pass

st.sidebar.caption(
    f"Chunk: {MIC_CHUNK_SEC}s · Whisper-tiny · {MIC_SAMPLE_RATE // 1000}kHz")

st.sidebar.divider()
st.sidebar.subheader("📚 Lecture RAG")
rag_api_key = st.sidebar.text_input(
    "NVIDIA API Key (optional)", type="password",
    help="Leave blank to view retrieved passages only. "
         "Provide NVIDIA NIM API key for full LLM answers.")
rag_llm_model = st.sidebar.selectbox(
    "LLM Model", [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ], index=0)

# =====================================================================
# MODEL INIT  (lazy, once per session)
# =====================================================================
if st.session_state.whisper_pipe is None:
    with st.spinner("Loading Whisper-tiny…"):
        st.session_state.whisper_pipe = load_whisper()

if st.session_state.hf_text_pipeline is None:
    with st.spinner("Loading text emotion model…"):
        st.session_state.hf_text_pipeline = load_hf_text_emotion_model()

if st.session_state.hf_face_pipeline is None or reload_face_model:
    with st.spinner("Loading face emotion model…"):
        st.session_state.hf_face_pipeline = load_hf_image_emotion_model(hf_face_name)

if st.session_state.gaze_bundle is None or gaze_checkpoint is not None:
    with st.spinner("Loading gaze model…"):
        st.session_state.gaze_bundle = load_gaze_model(
            vit_model_name, gaze_checkpoint if gaze_checkpoint else None)

if st.session_state.rag_embed_model is None and ST_AVAILABLE:
    st.session_state.rag_embed_model = load_embedding_model()

mp_face_mesh = _mp_face_mesh_mod
mp_drawing   = _mp_drawing_mod
_is_video_file = (video_source == "MP4 file")
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=_is_video_file,       # detect every frame for MP4; track for webcam
    refine_landmarks=True,
    max_num_faces=max_faces,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)

# =====================================================================
# START / STOP MIC + INFERENCE THREADS
# =====================================================================
mic_running = (st.session_state.audio_stop_evt is not None
               and not st.session_state.audio_stop_evt.is_set())

if start_monitor and not mic_running:
    st.session_state.audio_timeline        = []
    st.session_state.full_transcript       = ""
    st.session_state.latest_speech_emotion = None
    st.session_state.latest_speech_score   = 0.0
    st.session_state.audio_error           = None
    st.session_state.audio_result_q        = queue.Queue()
    st.session_state.audio_error_q         = queue.Queue()

    stop_evt = threading.Event()
    st.session_state.audio_stop_evt = stop_evt
    threading.Thread(
        target=_audio_worker,
        args=(stop_evt,
              st.session_state.whisper_pipe,
              st.session_state.hf_text_pipeline,
              st.session_state.audio_result_q,
              st.session_state.audio_error_q,
              mic_device_idx),
        daemon=True,
    ).start()
    st.sidebar.success("🎙️ Mic capture started")

    infer_stop = threading.Event()
    infer_in_q  = queue.Queue(maxsize=1)   # drop stale jobs automatically
    infer_out_q = queue.Queue()
    st.session_state.infer_stop_evt = infer_stop
    st.session_state.infer_in_q     = infer_in_q
    st.session_state.infer_out_q    = infer_out_q
    threading.Thread(
        target=_inference_worker,
        args=(infer_stop, infer_in_q, infer_out_q),
        daemon=True,
    ).start()

if not start_monitor and mic_running:
    st.session_state.audio_stop_evt.set()
    st.session_state.audio_stop_evt = None
    if st.session_state.infer_stop_evt:
        st.session_state.infer_stop_evt.set()
        st.session_state.infer_stop_evt = None
    st.sidebar.info("🎙️ Mic capture stopped")

if st.session_state.audio_error:
    st.sidebar.error(f"Mic error: {st.session_state.audio_error}")

# =====================================================================
# MAIN PAGE LAYOUT
# =====================================================================
st.title("🎓 Classroom Intelligence Suite")

tab_monitor, tab_rag = st.tabs(["📹 Live Monitoring", "📚 Lecture Q&A (RAG)"])

# =====================================================================
# RAG TAB
# =====================================================================
with tab_rag:
    st.markdown("### 📖 Upload a Lecture & Ask Questions")

    if not ST_AVAILABLE:
        st.error("**sentence-transformers** is not installed.\n\n"
                 "```\npip install sentence-transformers faiss-cpu\n```")
    elif not FAISS_AVAILABLE:
        st.error("**faiss-cpu** is not installed.\n\n"
                 "```\npip install faiss-cpu\n```")
    else:
        col_up, col_info = st.columns([2, 1])
        with col_up:
            lecture_file = st.file_uploader(
                "Upload lecture text (.txt, .md)", type=["txt", "md"],
                key="rag_lecture_up")
        with col_info:
            if st.session_state.rag_chunks:
                st.success(
                    f"✅ **{st.session_state.rag_lecture_name}** indexed\n\n"
                    f"📄 {len(st.session_state.rag_chunks)} chunks")
            else:
                st.info("No lecture indexed yet.")

        if lecture_file is not None:
            raw_text = lecture_file.read().decode("utf-8", errors="ignore")
            if lecture_file.name != st.session_state.rag_lecture_name:
                with st.spinner("🔄 Chunking & embedding lecture…"):
                    chunks, index = _build_rag_index(
                        raw_text, st.session_state.rag_embed_model)
                    st.session_state.rag_chunks       = chunks
                    st.session_state.rag_index        = index
                    st.session_state.rag_lecture_name = lecture_file.name
                    st.session_state.rag_chat_history = []
                st.success(
                    f"✅ Indexed **{len(chunks)} chunks** from *{lecture_file.name}*")

        st.divider()
        st.markdown("#### 💬 Ask a Question")

        if st.session_state.rag_chunks:
            for msg in st.session_state.rag_chat_history:
                icon = "🧑‍🎓" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"]):
                    st.markdown(f"{icon} {msg['content']}")

            user_q = st.chat_input("Ask something about the lecture…")
            if user_q:
                st.session_state.rag_chat_history.append(
                    {"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(f"🧑‍🎓 {user_q}")
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Retrieving relevant passages…"):
                        top_chunks = _retrieve_chunks(
                            user_q,
                            st.session_state.rag_chunks,
                            st.session_state.rag_index,
                            st.session_state.rag_embed_model)
                    with st.spinner("🤖 Generating answer…"):
                        answer = _rag_answer(
                            user_q, top_chunks,
                            api_key=rag_api_key, model=rag_llm_model)
                    st.markdown(f"🤖 {answer}")
                    with st.expander("📌 Retrieved source passages"):
                        for i, ch in enumerate(top_chunks, 1):
                            st.markdown(f"**Passage {i}:**\n> {ch}")
                st.session_state.rag_chat_history.append(
                    {"role": "assistant", "content": answer})

            if st.session_state.rag_chat_history:
                if st.button("🗑️ Clear Chat History", key="rag_clear"):
                    st.session_state.rag_chat_history = []
                    st.rerun()
        else:
            st.info("⬆️ Upload a lecture file above to get started.")

# =====================================================================
# MONITORING TAB
# =====================================================================
with tab_monitor:
    vid_col, metric_col = st.columns([3, 1])
    with vid_col:
        frame_window = st.image([], use_container_width=True)
    with metric_col:
        st.markdown("### 📊 Live Metrics")
        status_ph     = st.empty()
        speech_emo_ph = st.empty()
        last_tx_ph    = st.empty()
        st.markdown("---")
        st.markdown("### 📈 Attention")
        chart_ph = st.empty()

    timeline_ph = st.empty()

    # ── monitoring loop ────────────────────────────────────────────
    times_dq       = deque(maxlen=600)
    att_dq         = deque(maxlen=600)
    start_time_ref = time.time()

    if start_monitor:
        # open video source
        if video_source == "MP4 file":
            if uploaded_mp4 is None:
                st.warning("Please upload an MP4 file in the sidebar first.")
                st.stop()
            _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            _tmp.write(uploaded_mp4.read())
            _tmp.flush()
            cap = cv2.VideoCapture(_tmp.name)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open video source.")
            st.stop()

        PROCESS_EVERY  = 1.0
        FRAME_INTERVAL = 1.0 / 30.0       # cap render loop at ~30 fps
        last_proc_t    = time.time()
        last_log_t     = time.time()
        last_frame_t   = time.time()
        attention_level = 0
        last_faces_info = []
        gaze_model, gaze_device, gaze_tfm = st.session_state.gaze_bundle

        # Altair chart — incremental via add_rows()
        _att_chart = (
            alt.Chart(pd.DataFrame({"Time (s)": [], "Attention %": []}))
            .mark_line()
            .encode(
                x=alt.X("Time (s):Q"),
                y=alt.Y("Attention %:Q", scale=alt.Scale(domain=[0, 100])),
            )
            .properties(height=160)
        )
        chart_widget = chart_ph.altair_chart(_att_chart, use_container_width=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                if video_source == "MP4 file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    st.error("Video read failed.")
                    break

            now = time.time()

            # frame-rate cap
            if now - last_frame_t < FRAME_INTERVAL:
                time.sleep(0.002)
                continue
            last_frame_t = now

            h, w, _ = frame.shape

            # ── drain audio result queue ───────────────────────────
            _rq = st.session_state.get("audio_result_q")
            if _rq is not None:
                while not _rq.empty():
                    _entry = _rq.get_nowait()
                    st.session_state.latest_speech_emotion = _entry["emotion"]
                    st.session_state.latest_speech_score   = _entry["score"]
                    st.session_state.audio_timeline = (
                        st.session_state.audio_timeline + [_entry])
                    st.session_state.full_transcript = (
                        st.session_state.full_transcript + " " + _entry["text"]
                    ).strip()
                    _save_transcript_chunk(_entry["text"], class_name, subject_name)
            _eq = st.session_state.get("audio_error_q")
            if _eq is not None and not _eq.empty():
                st.session_state.audio_error = _eq.get_nowait()

            # ── per-second: submit crop batch to inference thread ──
            if (now - last_proc_t) >= PROCESS_EVERY:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                faces_rgb_list, bbox_list, lm_list = [], [], []
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks:
                        xs = [int(p.x * w) for p in lm.landmark]
                        ys = [int(p.y * h) for p in lm.landmark]
                        x0 = max(min(xs) - int(0.15 * (max(xs) - min(xs))), 0)
                        x1 = min(max(xs) + int(0.15 * (max(xs) - min(xs))), w - 1)
                        y0 = max(min(ys) - int(0.25 * (max(ys) - min(ys))), 0)
                        y1 = min(max(ys) + int(0.25 * (max(ys) - min(ys))), h - 1)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        crop = frame[y0:y1, x0:x1]
                        if crop.size == 0:
                            continue
                        faces_rgb_list.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        bbox_list.append((x0, y0, x1, y1))
                        lm_list.append(lm)

                _in_q = st.session_state.get("infer_in_q")
                if _in_q is not None and faces_rgb_list:
                    try:
                        _in_q.put_nowait((
                            faces_rgb_list, bbox_list, lm_list,
                            gaze_model, gaze_device, gaze_tfm,
                            st.session_state.hf_face_pipeline,
                            yaw_threshold, pitch_threshold, w, h,
                        ))
                    except queue.Full:
                        pass   # stale job still running — keep rendering
                last_proc_t = now

            # ── drain inference results ────────────────────────────
            _out_q = st.session_state.get("infer_out_q")
            if _out_q is not None:
                while not _out_q.empty():
                    _faces_info_raw = _out_q.get_nowait()
                    attentive_count = 0
                    total_faces     = len(_faces_info_raw)
                    new_faces_info  = []

                    # speech-face agreement modifier
                    _speech_emo = (st.session_state.latest_speech_emotion or "").lower()
                    SPEECH_ENGAGED    = {"joy", "surprise"}
                    SPEECH_DISENGAGED = {"sadness", "anger", "disgust", "fear"}
                    if _speech_emo in SPEECH_ENGAGED:
                        speech_category = "engaged"
                    elif _speech_emo in SPEECH_DISENGAGED:
                        speech_category = "disengaged"
                    else:
                        speech_category = "neutral"

                    FACE_ENGAGED    = {"happy", "surprise"}
                    FACE_DISENGAGED = {"sadness", "angry", "disgust", "fear"}

                    for fi in _faces_info_raw:
                        emo      = fi["emotion_label"]
                        gaze_lbl = fi["gaze"]

                        gaze_attentive = gaze_lbl in ["front", "up"]
                        emotion_attentive = (
                            False if emo in ("happy", "angry")
                            else gaze_lbl not in ["left", "right"]
                        )

                        face_category = (
                            "engaged"    if emo in FACE_ENGAGED    else
                            "disengaged" if emo in FACE_DISENGAGED else
                            "neutral"
                        )

                        if speech_category == "neutral" or (
                                speech_category == face_category == "neutral"):
                            emo_modifier = 0.0
                        elif speech_category == face_category:
                            emo_modifier = 0.15    # both agree → reinforce
                        else:
                            emo_modifier = -0.15   # conflict → reduce

                        combined = max(0.0, min(1.0,
                            (1.0 if (gaze_attentive or emotion_attentive) else 0.0)
                            + emo_modifier))
                        att_lbl = "Attentive" if combined >= 0.5 else "Distracted"
                        if att_lbl == "Attentive":
                            attentive_count += 1
                        color = (0, 255, 0) if att_lbl == "Attentive" else (0, 0, 255)
                        new_faces_info.append({
                            "bbox":             fi["bbox"],
                            "attention_label":  att_lbl,
                            "emotion_label":    emo,
                            "gaze":             gaze_lbl,
                            "color":            color,
                        })

                    last_faces_info = new_faces_info
                    attention_level = (
                        int((attentive_count / total_faces) * 100)
                        if total_faces else 0)
                    elapsed = int(now - start_time_ref)
                    times_dq.append(elapsed)
                    att_dq.append(attention_level)
                    chart_widget.add_rows(
                        pd.DataFrame({"Time (s)": [elapsed],
                                      "Attention %": [attention_level]}))
                    if (now - last_log_t) >= log_every:
                        _append_csv(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            elapsed, attention_level, class_name, subject_name)
                        last_log_t = now

            # ── draw face boxes ────────────────────────────────────
            for info in last_faces_info:
                x0, y0b, x1, y1b = info["bbox"]
                cv2.rectangle(frame, (x0, y0b), (x1, y1b), info["color"], 2)
                cv2.putText(
                    frame,
                    f"{info['attention_label']} | {info['emotion_label']} | {info['gaze']}",
                    (x0, max(0, y0b - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, info["color"], 2)

            # ── speech emotion banner (strip blend — no full copy) ─
            speech_emo = st.session_state.latest_speech_emotion
            if speech_emo:
                bgr   = EMOTION_COLORS.get(speech_emo.lower(), (180, 180, 180))
                score = st.session_state.latest_speech_score
                strip = frame[h - 44:h]
                bg    = np.full_like(strip, bgr, dtype=np.uint8)
                cv2.addWeighted(bg, 0.55, strip, 0.45, 0, strip)
                cv2.putText(frame,
                    f"SPEECH: {speech_emo.upper()}  ({score:.2f})",
                    (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

            # ── push frame ────────────────────────────────────────
            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                               use_container_width=True)

            # ── metric cards ──────────────────────────────────────
            distracted = 100 - attention_level
            status_ph.metric("👁️ Attention", f"{attention_level}%",
                             delta=f"{-distracted}% distracted")
            if speech_emo:
                speech_emo_ph.metric("🎙️ Speech Emotion",
                                     speech_emo.capitalize(),
                                     f"conf {score:.2f}")
            last_tx = st.session_state.full_transcript
            if last_tx:
                last_tx_ph.caption(f"📝 …{last_tx[-120:]}")

            # ── rolling speech timeline panel ─────────────────────
            tl = st.session_state.audio_timeline
            if tl:
                df_tl = pd.DataFrame(tl)
                with timeline_ph.container():
                    st.markdown("### 🎙️ Live Speech Emotion Timeline")
                    tc1, tc2 = st.columns([2, 1])
                    with tc1:
                        st.line_chart(df_tl.set_index("time")["score"],
                                      use_container_width=True)
                    with tc2:
                        st.dataframe(
                            df_tl[["time", "emotion", "score"]].tail(20).rename(
                                columns={"time": "Time(s)",
                                         "emotion": "Emotion", "score": "Score"}
                            ).style.format({"Score": "{:.2f}"}),
                            use_container_width=True, hide_index=True)
                    with st.expander("📝 Rolling Transcript"):
                        st.write(st.session_state.full_transcript or "—")

            if not start_monitor:
                break

        cap.release()

    else:
        # ── idle placeholder ───────────────────────────────────────
        idle = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(idle, "Toggle 'Start Monitoring' in the sidebar to begin",
                    (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 2)
        frame_window.image(idle, use_container_width=True)
        status_ph.info("Waiting for session to start…")

        tl = st.session_state.audio_timeline
        if tl:
            df_tl = pd.DataFrame(tl)
            with timeline_ph.container():
                st.markdown("### 🎙️ Speech Emotion Timeline")
                tc1, tc2 = st.columns([2, 1])
                with tc1:
                    st.line_chart(df_tl.set_index("time")["score"],
                                  use_container_width=True)
                with tc2:
                    st.dataframe(
                        df_tl[["time", "emotion", "score"]].tail(20).rename(
                            columns={"time": "Time(s)",
                                     "emotion": "Emotion", "score": "Score"}
                        ).style.format({"Score": "{:.2f}"}),
                        use_container_width=True, hide_index=True)
                with st.expander("📝 Rolling Transcript"):
                    st.write(st.session_state.full_transcript or "—")