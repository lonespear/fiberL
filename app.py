import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from fiberL import fiberL  # Ensure your fiberL code is saved as 'fiberL_module.py'
import tempfile
import shutil
from io import BytesIO

st.set_page_config(page_title="Fiber Length Analysis", layout="wide")
st.title("🧪 Fiber Length Analysis using fiberL")

st.sidebar.header("📸 Upload SEM Image")
# --- Image Upload + Session State ---
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None

new_file = st.sidebar.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

# Reset session if user removes image
if new_file is None and st.session_state.uploaded_file is not None:
    st.session_state.uploaded_file = None
    st.session_state.cropped_img = None
    st.rerun()

# If user uploads a new file
if new_file is not None:
    st.session_state.uploaded_file = new_file
    uploaded_file = new_file
else:
    uploaded_file = None

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.format = "TIFF"  # Set format manually so st_canvas doesn't crash


    st.sidebar.subheader("📐 Scale Units")
    unit_options = ["microns", "nanometers", "millimeters", "inches"]
    selected_unit = st.sidebar.selectbox("Choose unit of measurement", unit_options, index=0)

    st.subheader("📏 Step 1: Measure Scale Bar")

    scale_canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=img_pil,
        update_streamlit=True,
        height=img_pil.height,
        width=img_pil.width,
        drawing_mode="line",
        key="scale_bar"
    )

    if scale_canvas.json_data is not None:
        objs = scale_canvas.json_data["objects"]
        if objs and objs[0]["type"] == "line":
            x1, y1 = objs[0]["x"], objs[0]["y"]
            x2, y2 = x1 + objs[0]["width"], y1 + objs[0]["height"]
            pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])
            st.write(f"🧮 Pixel Length: `{pixel_distance:.2f}`")

            real_length = st.number_input(f"Enter the real-world length of the scale bar (in {selected_unit})", min_value=0.0001)
            if real_length:
                pixels_per_unit = pixel_distance / real_length
                st.session_state["pixels_per_unit"] = pixels_per_unit
                st.session_state["unit_label"] = selected_unit

    if st.session_state.cropped_img is None:
        st.subheader("🖼️ Crop Region of Interest")
        rect = st_cropper(img_pil, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
        cropped_img = np.array(rect)

        if st.button("📸 Confirm Crop"):
            st.session_state.cropped_img = cropped_img
            st.rerun()
    else:
        cropped_img = st.session_state.cropped_img

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

