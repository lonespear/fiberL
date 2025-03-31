import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates
from fiberL import fiberL

st.set_page_config(page_title="Fiber Length Analysis", layout="wide")
st.title("🧪 Fiber Length Analysis using fiberL")

st.sidebar.header("📸 Upload SEM Image")

# --- Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'points' not in st.session_state:
    st.session_state.points = []

uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_original = Image.fromarray(img_rgb)

    # Resize for display (not processing)
    max_display_width = 700
    w_percent = max_display_width / pil_original.width
    new_height = int(pil_original.height * w_percent)
    pil_resized = pil_original.resize((max_display_width, new_height))
    scale_ratio = pil_original.width / max_display_width

    st.sidebar.subheader("📐 Scale Units")
    unit_options = ["microns", "nanometers", "millimeters", "inches"]
    selected_unit = st.sidebar.selectbox("Choose unit of measurement", unit_options, index=0)

    st.markdown("## ⚙️ Workflow Options")
    scale_toggle = st.toggle("Enable Scale Measurement", value=True)
    crop_toggle = st.toggle("Enable Cropping", value=True)

    # --- SCALE MEASUREMENT ---
    if scale_toggle:
        st.subheader("📏 Step 1: Measure Scale Bar (click two ends)")
        coords = streamlit_image_coordinates(pil_resized)

        if coords and len(st.session_state.points) < 2:
            scaled_point = (coords['x'] * scale_ratio, coords['y'] * scale_ratio)
            st.session_state.points.append(scaled_point)
            st.write(f"Point {len(st.session_state.points)}: ({scaled_point[0]:.1f}, {scaled_point[1]:.1f})")

        annotated = pil_original.copy()
        draw = ImageDraw.Draw(annotated)

        if len(st.session_state.points) >= 1:
            x1, y1 = st.session_state.points[0]
            draw.ellipse([(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)], fill='red')
            draw.text((x1 + 6, y1), "A", fill='red')

        if len(st.session_state.points) == 2:
            x1, y1 = st.session_state.points[0]
            x2, y2 = st.session_state.points[1]
            draw.ellipse([(x2 - 5, y2 - 5), (x2 + 5, y2 + 5)], fill='blue')
            draw.text((x2 + 6, y2), "B", fill='blue')
            draw.line([x1, y1, x2, y2], fill='yellow', width=2)

            pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])
            st.image(annotated, caption="📏 Annotated Scale Bar", use_container_width=True)
            st.write(f"🧮 Pixel Distance: `{pixel_distance:.2f}`")

            real_length = st.number_input(
                f"Enter real-world length of this line (in {selected_unit})", min_value=0.0001
            )
            if real_length:
                pixels_per_unit = pixel_distance / real_length
                st.session_state["pixels_per_unit"] = pixels_per_unit
                st.session_state["unit_label"] = selected_unit

            if st.button("🔁 Reset Measurement"):
                st.session_state.points = []
        elif len(st.session_state.points) == 1:
            st.image(annotated, caption="🖱️ Click one more point to complete the scale line", use_container_width=True)

        if len(st.session_state.points) < 2:
            st.stop()  # Don't continue to cropping unless both points are selected

    # --- CROPPING ---
    if crop_toggle:
        if st.session_state.cropped_img is None:
            st.subheader("🖼️ Crop Region of Interest")
            rect = st_cropper(pil_original, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
            cropped_img = np.array(rect)
            if st.button("📸 Confirm Crop"):
                st.session_state.cropped_img = cropped_img
                st.rerun()
        else:
            cropped_img = st.session_state.cropped_img
    else:
        cropped_img = np.array(pil_original)

    # --- PREPROCESSING + ANALYSIS ---
    st.sidebar.header("⚙️ Preprocessing Parameters")
    niter = st.sidebar.slider("Anisotropic Diffusion Iterations", 1, 100, 50)
    kappa = st.sidebar.slider("Kappa (Conduction Coefficient)", 1, 100, 50)
    gamma = st.sidebar.slider("Gamma (Diffusion Rate)", 0.01, 0.25, 0.2)
    thresh_1 = st.sidebar.slider("Initial Binary Threshold", 0, 255, 126)
    g_blur = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 31, 9, step=2)
    thresh_2 = st.sidebar.slider("Final Binary Threshold", 0, 255, 15)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Algorithm Parameters")

    ksize = st.sidebar.slider("Kernel Size for Branch Point Dilation", 1, 15, 5)
    min_prune = st.sidebar.slider("Minimum Edge Prune Length", 1, 50, 5)
    max_node_dist = st.sidebar.slider("Branch Merge Distance Threshold", 1, 50, 15)
    tip_distance_thresh = st.sidebar.slider("Maximum Tip Distance Threshold", 0, 50, 25)
    cos_thresh = st.sidebar.slider("Tip Merging Cosine Threshold", 0.0, 1.0, 0.85)
    curvature_thresh = st.sidebar.slider("Curvature Similarity Threshold", 0.0, 1.0, 0.85)

    st.subheader("🔍 Skeletonization Preview")
    analyzer = fiberL(
        image=cropped_img.copy(),
        niter=niter,
        kappa=kappa,
        gamma=gamma,
        thresh_1=thresh_1,
        g_blur=g_blur,
        thresh_2=thresh_2,
        ksize=ksize,
        min_prune=min_prune,
        max_node_dist=max_node_dist,
        cos_thresh=cos_thresh,
        curvature_thresh=curvature_thresh,
        pixels_per_unit=st.session_state.get("pixels_per_unit", 1.0)
    )
    analyzer.preproc()

    col1, col2 = st.columns(2)
    with col1:
        st.image(cropped_img, caption="Cropped Original Image", use_container_width=True)
    with col2:
        st.image(analyzer.sk_image * 255, caption="Skeletonized Image", use_container_width=True, clamp=True)

    if st.button("🚀 Run Full Fiber Length Analysis"):
        with st.spinner("Processing image... This may take a moment..."):
            analyzer.find_length()
            st.success("✅ Analysis Complete!")

            st.subheader("🎨 Final Colored Edge Network")
            st.image(analyzer.color_image, channels="RGB", use_container_width=True)

            st.subheader("📊 Fiber Length Histogram")
            st.pyplot(analyzer.fig)

            st.subheader("📊 Summary Statistics")
            st.dataframe(analyzer.stats_df.style.format({"Value": "{:.2f}"}))

            with st.expander("📤 Export Results"):
                analyzer.export_results_streamlit()
                st.success("Results exported to fiberL_output folder!")
