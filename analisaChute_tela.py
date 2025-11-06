import streamlit as st
from analisaChute import AnalisaChutes
import tempfile
import os
import time
import numpy as np
import cv2

# Configs
RECHECK_COLOR_EVERY = 10   # reavaliar cor a cada N frames
COLOR_MATCH_THRESHOLD = 0.2  # % de pixels do crop que devem bater com a cor alvo
RESIZE_W, RESIZE_H = 540, 360
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS

def get_main_color_score(image, bbox, target="vermelha"):
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1]-1, x_max), min(image.shape[0]-1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    if target == "vermelha":
        lower1 = np.array([0, 120, 70]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70]); upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([90, 50, 50]); upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    nonzero = int(np.count_nonzero(mask))
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return nonzero / total  # score entre 0 e 1

def get_bbox_from_landmarks(landmarks, image_shape):
    idxs = [11, 12, 23, 24]  # ombros e quadris
    width = image_shape[1]; height = image_shape[0]
    xs = []; ys = []
    for i in idxs:
        if i < len(landmarks):
            x = landmarks[i][0]; y = landmarks[i][1]
            if x is None or y is None:
                continue
            xs.append(int(x * width)); ys.append(int(y * height))
    if not xs or not ys:
        return 0, 0, width-1, height-1
    x_min, x_max = max(min(xs), 0), min(max(xs), width-1)
    y_min, y_max = max(min(ys), 0), min(max(ys), height-1)
    pad_w = int((x_max - x_min) * 0.2) or 2
    pad_h = int((y_max - y_min) * 0.3) or 2
    return max(0, x_min - pad_w), max(0, y_min - pad_h), min(width-1, x_max + pad_w), min(height-1, y_max + pad_h)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Análise de Chutes")

# session_state defaults
st.session_state.setdefault("video_path", None)
st.session_state.setdefault("analisachutes", None)
st.session_state.setdefault("started", False)
st.session_state.setdefault("frame_count", 0)
st.session_state.setdefault("selected_idx", None)
st.session_state.setdefault("prev_cor_alvo", None)
st.session_state.setdefault("force_color_recheck", False)
st.session_state.setdefault("last_display_frame", None)  # <-- novo: persiste frame exibido entre reruns
# Placeholders
placeholder_angulos = st.empty()
placeholder = st.empty()

# Controls
col1, col2, col3, col4 = st.sidebar.columns(4)
with col1:
    start_btn = st.button("▶ Start")
with col2:
    pause_btn = st.button("⏸ Pause")
with col3:
    resume_btn = st.button("▶ Resume")
with col4:
    stop_btn = st.button("⏹ Stop")

cor_alvo = st.sidebar.selectbox(
    "Selecione a cor da roupa a ser analisada:",
    ("vermelha", "azul"),
    index=0
)

# When color changes, force re-evaluation (do NOT recreate video)
if st.session_state.prev_cor_alvo is None:
    st.session_state.prev_cor_alvo = cor_alvo
elif cor_alvo != st.session_state.prev_cor_alvo:
    st.session_state.selected_idx = None
    st.session_state.frame_count = 0
    st.session_state.force_color_recheck = True
    st.session_state.prev_cor_alvo = cor_alvo

# Upload
arquivo = st.sidebar.file_uploader("📂 Importar Vídeo", type=["mp4", "avi", "mov", "mkv"])
if arquivo is not None:
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, arquivo.name)
    with open(temp_path, "wb") as f:
        f.write(arquivo.read())
    # if changed file, stop previous and set new path (do not auto-run)
    if st.session_state.video_path != temp_path:
        st.session_state.video_path = temp_path
        if st.session_state.analisachutes is not None:
            try:
                st.session_state.analisachutes.stop()
            except Exception:
                pass
            st.session_state.analisachutes = None
        st.success("Vídeo carregado. Clique Start para iniciar o processamento.")
else:
    if st.session_state.video_path is None:
        st.info("Selecione um vídeo")

# Button actions
if start_btn:
    if st.session_state.video_path is None:
        st.warning("Carregue um vídeo primeiro.")
    else:
        if st.session_state.analisachutes is None:
            st.session_state.analisachutes = AnalisaChutes(st.session_state.video_path)
            # start processing thread (no drawing for perf)
            try:
                st.session_state.analisachutes.run(False)
            except TypeError:
                # compatibility if signature is different
                st.session_state.analisachutes.run(False, False)
        else:
            # if instance exists but not running, start; if paused, resume
            if not getattr(st.session_state.analisachutes, "running", False):
                st.session_state.analisachutes.run(False)
            elif getattr(st.session_state.analisachutes, "paused", False):
                st.session_state.analisachutes.resume()
        st.session_state.started = True

if pause_btn and st.session_state.analisachutes is not None:
    st.session_state.analisachutes.pause()

if resume_btn and st.session_state.analisachutes is not None:
    st.session_state.analisachutes.resume()

if stop_btn and st.session_state.analisachutes is not None:
    st.session_state.analisachutes.stop()
    st.session_state.analisachutes = None
    st.session_state.started = False
    st.session_state.selected_idx = None

# Aux vars
pernaD = 0.0; pernaE = 0.0
last_frame = None
count = 0
status = "repouso"
preparado = "pronto"
ChutesEfetivos = 0

# Ensure counters
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

# Main loop: only while pipeline exists and is running
while st.session_state.analisachutes is not None and getattr(st.session_state.analisachutes, "running", False):
    loop_start = time.time()
    st.session_state.frame_count += 1

    # If paused, just show last frame and wait
    if getattr(st.session_state.analisachutes, "paused", False):
        disp = st.session_state.get("last_display_frame")
        # fallback para o last_frame local (redimensionar para exibição)
        if disp is None and last_frame is not None:
            try:
                if isinstance(last_frame, np.ndarray):
                    disp = cv2.resize(last_frame, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
                    st.session_state["last_display_frame"] = disp
                else:
                    disp = None
            except Exception:
                disp = None
        if disp is not None and isinstance(disp, np.ndarray):
            placeholder.image(disp, channels="RGB")
        # show persisted angles/count so pause preserves values
        pernaD_disp = st.session_state.get("pernaD", pernaD)
        pernaE_disp = st.session_state.get("pernaE", pernaE)
        chutes_disp = st.session_state.get("ChutesEfetivos", ChutesEfetivos)
        placeholder_angulos.write(f"Perna Direita: {pernaD_disp:.1f}° | Perna Esquerda: {pernaE_disp:.1f}° | Contagem: {ChutesEfetivos}")
        time.sleep(FRAME_TIME)
        continue

    item = st.session_state.analisachutes.image_q.get()
    if not item:
        time.sleep(FRAME_TIME); continue
    frame, landmarks, all_people, ts = item
    last_frame = frame

    # Handle end marker
    if isinstance(frame, str) and frame == "END":
        if last_frame is not None and isinstance(last_frame, np.ndarray):
            placeholder.image(last_frame, channels="RGB")
        st.write("Fim do vídeo.")
        # cleanup
        try:
            st.session_state.analisachutes.stop()
        except Exception:
            pass
        st.session_state.analisachutes = None
        st.session_state.started = False
        break

    # Resize/process only if valid frame
    if isinstance(frame, np.ndarray):
        try:
            proc = cv2.resize(frame, (RESIZE_W, RESIZE_H))
            
            # salva o frame redimensionado exibido para que Pause o use (persistente entre reruns)
            st.session_state["last_display_frame"] = proc
            last_frame = proc
            
        except Exception:
            time.sleep(FRAME_TIME)
            continue
    else:
        time.sleep(FRAME_TIME); continue

    # Validate landmarks/pessoas
    if landmarks is None or not isinstance(all_people, (list, tuple)) or len(all_people) == 0:
        time.sleep(FRAME_TIME); continue

    # Re-evaluate selection
    if st.session_state.selected_idx is None or st.session_state.frame_count % RECHECK_COLOR_EVERY == 0 or st.session_state.force_color_recheck:
        best_idx = None; best_score = -0.5
        frame_np = proc if isinstance(proc, np.ndarray) else np.array(proc)
        for idx, person_landmarks in enumerate(all_people):
            try:
                bbox = get_bbox_from_landmarks(person_landmarks, frame_np.shape)
            except Exception:
                continue
            score = get_main_color_score(frame_np, bbox, target=cor_alvo)
            if score > best_score:
                best_score = score; best_idx = idx
        if best_score >= COLOR_MATCH_THRESHOLD:
            st.session_state.selected_idx = best_idx
        else:
            if st.session_state.selected_idx is None:
                st.session_state.selected_idx = best_idx
        st.session_state.force_color_recheck = False

    selected_idx = st.session_state.selected_idx

    # validate index
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(all_people):
        time.sleep(FRAME_TIME); continue

    # compute angles for selected person
    try:
        _, pernaD = st.session_state.analisachutes.find_angle(proc, landmarks, 26, 24, 23, False, person_idx=selected_idx)
        _, pernaE = st.session_state.analisachutes.find_angle(proc, landmarks, 24, 23, 25, False, person_idx=selected_idx)
    except Exception:
        time.sleep(FRAME_TIME); continue

    if pernaD is None or pernaE is None:
        time.sleep(FRAME_TIME); continue

    # PERSISTIR os últimos ângulos e contadores para que Pause mostre valores corretos
    st.session_state["pernaD"] = pernaD
    st.session_state["pernaE"] = pernaE
    st.session_state.setdefault("ChutesEfetivos", ChutesEfetivos)
    st.session_state.setdefault("count", count)


    if (pernaD < -50 and pernaE > 200) or (pernaE < -50 and pernaD > 200):
        status = "repouso"; preparado = "pronto"

    if status == "repouso":
        if (preparado == "pronto" and pernaD > 170 and pernaE >= 200) or (pernaE > 170 and pernaD >= 200)or (preparado == "pronto" and pernaD < -80 and pernaE <= -100) or (pernaE < -80 and pernaD <= -100):
                
            preparado = "chutando"; count += 1
    if count >= 23:
        ChutesEfetivos += 1; count = 0

    with placeholder.container():
        col_a, col_b = st.columns([2, 1])
        col_a.image(proc, channels="RGB")
        col_b.markdown(f"### **Contagem**: {ChutesEfetivos}")

    placeholder_angulos.write(f"Perna Direita: {pernaD:.1f}° | Perna Esquerda: {pernaE:.1f}° | Selecionado: {selected_idx}")

    elapsed = time.time() - loop_start
    time.sleep(max(0, FRAME_TIME - elapsed))
