import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import time
from pathlib import Path

from ultralytics import YOLO

# PDF Report Generation
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import TableStyle

# ─── Hardcoded Model Path ─────────────────────────────────────────
MODEL_PATH = Path(r"D:\AI SPRY projects\AI-Based Pothole Severity Assessment System\best.pt")

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PotholeVision",
    page_icon="🕳",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #12121a;
    --border:  #1e1e2e;
    --accent:  #f5c542;
    --accent2: #ff6b35;
    --text:    #e8e8f0;
    --muted:   #6b6b80;
    --small:   #4ade80;
    --medium:  #facc15;
    --large:   #f87171;
}
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }
.hero { display: flex; align-items: center; gap: 1.2rem; padding: 2rem 0 0.5rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.hero-icon { font-size: 3.2rem; line-height: 1; filter: drop-shadow(0 0 16px #f5c54266); }
.hero-title { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; line-height: 1.1; }
.hero-sub { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--muted); margin: 0.3rem 0 0; letter-spacing: 0.08em; text-transform: uppercase; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
.upload-hint { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); text-align: center; padding: 0.5rem; letter-spacing: 0.06em; }
.metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem 1.4rem; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.metric-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.metric-value { font-size: 2rem; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-unit { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; font-family: 'Space Mono', monospace; }
.styled-table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin: 1.5rem 0; }
.styled-table-header { font-family: 'Space Mono', monospace; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); padding: 1rem 1.5rem 0.6rem; border-bottom: 1px solid var(--border); }
.dataframe { background: transparent !important; border: none !important; font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important; }
.dataframe th { background: #1a1a26 !important; color: var(--muted) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; border: none !important; padding: 0.7rem 1rem !important; }
.dataframe td { background: transparent !important; color: var(--text) !important; border-color: var(--border) !important; padding: 0.65rem 1rem !important; }
.dataframe tr:hover td { background: #1a1a2a !important; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 700; font-family: 'Space Mono', monospace; letter-spacing: 0.05em; text-transform: uppercase; }
.badge-small  { background: #14532d; color: var(--small);  border: 1px solid #166534; }
.badge-medium { background: #713f12; color: var(--medium); border: 1px solid #854d0e; }
.badge-large  { background: #7f1d1d; color: var(--large);  border: 1px solid #991b1b; }
.section-label { font-family: 'Space Mono', monospace; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); margin: 2rem 0 0.8rem; display: flex; align-items: center; gap: 0.6rem; }
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.status-pill { display: inline-flex; align-items: center; gap: 0.5rem; background: #0d2218; border: 1px solid #166534; color: var(--small); border-radius: 100px; padding: 0.3rem 0.9rem; font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em; }
.status-dot { width: 7px; height: 7px; background: var(--small); border-radius: 50%; animation: pulse 1.5s ease infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }
.error-box { background: #1a0808; border: 1px solid #991b1b; border-left: 3px solid var(--large); border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--large); margin: 1rem 0; }
.info-box { background: #12121a; border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-family: 'Space Mono', monospace; font-size: 0.75rem; color: var(--muted); margin: 1rem 0; }
.stButton > button { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #0a0a0f; border: none; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.03em; padding: 0.6rem 1.6rem; border-radius: 8px; transition: opacity 0.2s, transform 0.15s; width: 100%; }
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
.stSlider > div > div > div { background: var(--accent) !important; }
[data-testid="stFileUploader"] { background: var(--surface); border: 1.5px dashed var(--border); border-radius: 12px; padding: 1rem; transition: border-color 0.2s; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 8px; padding: 4px; gap: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); border-radius: 6px; padding: 0.5rem 1rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--accent)22, var(--accent2)22) !important; color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🕳</div>
    <div>
        <div class="hero-title">Pothole Severity Assessment</div>
        <div class="hero-sub">AI-Powered Road Defect Detection &amp; Severity Analysis · YOLOv8n-seg</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def classify_pothole(area_px, img_area):
    r = area_px / img_area
    return "Small" if r < 0.01 else ("Medium" if r < 0.04 else "Large")

# dynamic road width
def area_to_sqm(area_px, img_w, road_width_m):
    return round(area_px / (img_w / road_width_m) ** 2, 4)

SIZE_COLORS = {"Small": (74, 222, 128),
               "Medium": (250, 204, 21),
               "Large": (248, 113, 113)}

# road_width_m parameter
def run_inference(model, img_bgr, conf, iou, road_width_m):
    results  = model.predict(img_bgr, conf=conf, iou=iou, verbose=False)
    r        = results[0]
    h, w     = img_bgr.shape[:2]
    img_area = h * w
    rows     = []
    annotated = img_bgr.copy()

    if r.masks is not None:
        for i, mask in enumerate(r.masks.data.cpu().numpy()):
            mask_bin = (cv2.resize(mask, (w, h)) > 0.5).astype(np.uint8)
            area_px  = int(mask_bin.sum())
            size     = classify_pothole(area_px, img_area)
            color    = SIZE_COLORS[size]
            conf_val = float(r.boxes.conf[i])

            overlay = annotated.copy()
            overlay[mask_bin == 1] = color
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
            label = f"#{i+1} {size} {conf_val:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1+4, y1-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (10,10,20), 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            rows.append({
                "ID": f"#{i+1}",
                "Confidence": f"{conf_val:.1%}",
                "Area (px²)": area_px,
                "Area (m²)":  area_to_sqm(area_px, w, road_width_m),
                "Size Class": size,
            })

    return annotated, rows


def render_results_table(rows):
    if not rows:
        st.warning("No potholes detected. Try lowering the confidence threshold.")
        return

    df = pd.DataFrame(rows)
    df_display = df.copy()
    df_display["Size Class"] = df_display["Size Class"].apply(
        lambda s: f'<span class="badge badge-{s.lower()}">{s}</span>')

    st.markdown('<div class="styled-table-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="styled-table-header">Detection Results</div>', unsafe_allow_html=True)
    st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    total  = len(rows)
    small  = sum(1 for r in rows if r["Size Class"] == "Small")
    medium = sum(1 for r in rows if r["Size Class"] == "Medium")
    large  = sum(1 for r in rows if r["Size Class"] == "Large")

    # Risk Score Calculation
    risk_score = small*1 + medium*3 + large*5

    if risk_score == 0:
        risk_level = "SAFE"
        priority = "No Immediate Action"
    elif risk_score <= 5:
        risk_level = "MODERATE"
        priority = "Scheduled Maintenance"
    elif risk_score <= 15:
        risk_level = "HIGH"
        priority = "High Priority Repair"
    else:
        risk_level = "CRITICAL"
        priority = "Urgent Intervention Required"

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card"><div class="metric-label">Total</div>
            <div class="metric-value">{total}</div>
            <div class="metric-unit">potholes</div></div>
        <div class="metric-card"><div class="metric-label">Small</div>
            <div class="metric-value" style="color:var(--small)">{small}</div></div>
        <div class="metric-card"><div class="metric-label">Medium</div>
            <div class="metric-value" style="color:var(--medium)">{medium}</div></div>
        <div class="metric-card"><div class="metric-label">Large</div>
            <div class="metric-value" style="color:var(--large)">{large}</div></div>
    </div>
    """, unsafe_allow_html=True)

    #  Risk Card 
    st.markdown(f"""
    <div class="metric-card" style="margin-top:1rem;">
        <div class="metric-label">Road Risk Index</div>
        <div class="metric-value">{risk_level}</div>
        <div class="metric-unit">Score: {risk_score} · {priority}</div>
    </div>
    """, unsafe_allow_html=True)

    # store analytics
    st.session_state["analytics_data"] = rows
    st.session_state["risk_level"] = risk_level
    st.session_state["risk_score"] = risk_score
    st.session_state["priority"] = priority

    st.download_button("⬇ Download Results CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="pothole_results.csv",
                       mime="text/csv")
    
# ─── Load Model Once (cached) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = None

if MODEL_PATH.exists():
    with st.spinner("Loading YOLOv8n-seg weights..."):
        try:
            model = load_model(str(MODEL_PATH))
            st.markdown(
                '<div class="status-pill"><span class="status-dot"></span> YOLOv8n-seg ready</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.markdown(
                f'<div class="error-box">❌ Failed to load model: {e}</div>',
                unsafe_allow_html=True
            )
else:
    st.markdown(
        f'<div class="error-box">❌ best.pt not found at {MODEL_PATH}</div>',
        unsafe_allow_html=True
    )
    
# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown("---")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.25, 0.05)
    iou_thresh  = st.slider("IoU Threshold",         0.10, 0.95, 0.45, 0.05)

    st.markdown("---")

    # 🔥 NEW SECTION (does NOT disturb layout)
    st.markdown("### 🛣 Road Settings")
    road_width_m = st.slider("Estimated Road Width (meters)", 2.5, 7.0, 3.5, 0.5)

    st.markdown("---")

    st.markdown("### 📐 Size Classification")
    st.markdown("""
    <div class="info-box">
        <span style="color:var(--small)">● Small</span>  — area &lt; 1%<br>
        <span style="color:var(--medium)">● Medium</span> — area 1%–4%<br>
        <span style="color:var(--large)">● Large</span>  — area &gt; 4%
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📹 Video Settings")
    frame_skip = st.slider("Process every Nth frame", 1, 10, 3)
    max_frames = st.slider("Max frames to process",  30, 500, 150)

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:var(--muted);text-align:center;">
        PotholeVision v1.0<br>
        YOLOv8n-seg · 9,052 images<br>
        Box mAP50: 0.742 · Mask mAP50: 0.734
    </div>""", unsafe_allow_html=True)
    
def generate_pdf_report(rows, risk_level, risk_score, priority):
    filename = "maintenance_report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("PotholeVision Maintenance Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {risk_score}", styles["Normal"]))
    elements.append(Paragraph(f"Priority: {priority}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    data = [["ID", "Confidence", "Area (m²)", "Size Class"]]
    for r in rows:
        data.append([r["ID"], r["Confidence"], r["Area (m²)"], r["Size Class"]])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black)
    ]))

    elements.append(table)
    doc.build(elements)

    return filename
    
# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_analytics = st.tabs(
    ["📷  Image Detection", "🎬  Video Detection", "📊  Analytics"]
)

# IMAGE TAB
with tab_img:
    st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader(
        "img",
        type=["jpg","jpeg","png","bmp","webp"],
        key="img_upload",
        label_visibility="collapsed"
    )
    st.markdown('<div class="upload-hint">Supported: JPG · PNG · BMP · WEBP</div>',
                unsafe_allow_html=True)

    if uploaded_img:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        col_orig, col_result = st.columns(2, gap="large")

        with col_orig:
            st.markdown('<div class="section-label">Original</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width="stretch")
            h, w = img_bgr.shape[:2]
            st.caption(f"Resolution: {w} × {h} px")

        if model is None:
            st.error("Model not loaded — place best.pt next to app.py.")
        else:
            with col_result:
                st.markdown('<div class="section-label">Detection Output</div>',
                            unsafe_allow_html=True)
                with st.spinner("Running inference..."):
                    t0 = time.time()
                    annotated, rows = run_inference(
                        model,
                        img_bgr,
                        conf_thresh,
                        iou_thresh,
                        road_width_m
                    )
                    elapsed = time.time() - t0

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
                st.caption(
                    f"Inference: {elapsed*1000:.0f} ms · {len(rows)} pothole(s) detected"
                )

            st.markdown('<div class="section-label">Results Table</div>',
                        unsafe_allow_html=True)

            render_results_table(rows)

            # PDF Download
            if rows and "risk_level" in st.session_state:
                pdf_file = generate_pdf_report(
                    rows,
                    st.session_state["risk_level"],
                    st.session_state["risk_score"],
                    st.session_state["priority"]
                )

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "⬇ Download Maintenance Report (PDF)",
                        f.read(),
                        file_name="maintenance_report.pdf",
                        mime="application/pdf"
                    )

            # Store analytics
            st.session_state["analytics_data"] = rows

            _, buf = cv2.imencode(".jpg", annotated)
            st.download_button(
                "⬇ Download Annotated Image",
                buf.tobytes(),
                file_name="pothole_annotated.jpg",
                mime="image/jpeg"
            )


# VIDEO TAB
with tab_vid:
    st.markdown('<div class="section-label">Upload Video</div>', unsafe_allow_html=True)
    uploaded_vid = st.file_uploader(
        "vid",
        type=["mp4","avi","mov","mkv"],
        key="vid_upload",
        label_visibility="collapsed"
    )
    st.markdown('<div class="upload-hint">Supported: MP4 · AVI · MOV · MKV</div>',
                unsafe_allow_html=True)

    if uploaded_vid:
        if model is None:
            st.error("Model not loaded — place best.pt next to app.py.")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
            w_vid   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            st.markdown(f"""
            <div class="info-box">
                {w_vid}×{h_vid} px · {total_frames} frames ·
                {fps_vid:.1f} FPS · {total_frames/fps_vid:.1f}s
            </div>""", unsafe_allow_html=True)

            if st.button("▶ Run Detection on Video"):
                cap      = cv2.VideoCapture(tfile.name)
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out_vid  = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps_vid / frame_skip,
                    (w_vid, h_vid)
                )

                all_rows, frame_idx, processed = [], 0, 0
                preview_slot = st.empty()
                progress_bar = st.progress(0)
                status_text  = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret or processed >= max_frames:
                        break

                    if frame_idx % frame_skip == 0:
                        ann_f, rows_f = run_inference(
                            model,
                            frame,
                            conf_thresh,
                            iou_thresh,
                            road_width_m
                        )

                        for r in rows_f:
                            r["Frame"] = frame_idx

                        all_rows.extend(rows_f)
                        out_vid.write(ann_f)

                        if processed % 5 == 0:
                            preview_slot.image(
                                cv2.cvtColor(ann_f, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_idx} · {len(rows_f)} pothole(s)",
                                width="stretch"
                            )

                        processed += 1
                        progress_bar.progress(min(processed / max_frames, 1.0))
                        status_text.text(
                            f"Frame {frame_idx} ({processed}/{max_frames})..."
                        )

                    frame_idx += 1

                cap.release()
                out_vid.release()

                progress_bar.progress(1.0)
                status_text.text(
                    f"✅ Done — {processed} frames · {len(all_rows)} total detections"
                )

                st.markdown('<div class="section-label">Aggregated Results</div>',
                            unsafe_allow_html=True)

                render_results_table(all_rows)

                # PDF Download (Video)
                if all_rows and "risk_level" in st.session_state:
                    pdf_file = generate_pdf_report(
                        all_rows,
                        st.session_state["risk_level"],
                        st.session_state["risk_score"],
                        st.session_state["priority"]
                    )

                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            "⬇ Download Maintenance Report (PDF)",
                            f.read(),
                            file_name="maintenance_report.pdf",
                            mime="application/pdf"
                        )

                # Store analytics
                st.session_state["analytics_data"] = all_rows

                with open(out_path, "rb") as vf:
                    st.download_button(
                        "⬇ Download Annotated Video",
                        vf.read(),
                        file_name="pothole_annotated.mp4",
                        mime="video/mp4"
                    )

                os.unlink(tfile.name)


# ANALYTICS TAB
with tab_analytics:
    st.markdown('<div class="section-label">System Analytics</div>',
                unsafe_allow_html=True)

    if "analytics_data" not in st.session_state:
        st.info("Run detection first to generate analytics.")
    else:
        df = pd.DataFrame(st.session_state["analytics_data"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Severity Distribution")
            st.bar_chart(df["Size Class"].value_counts())

        with col2:
            st.subheader("Confidence Distribution")
            conf_vals = df["Confidence"].str.replace("%","").astype(float)
            st.line_chart(conf_vals)

        st.subheader("Area Distribution (m²)")
        st.area_chart(df["Area (m²)"])