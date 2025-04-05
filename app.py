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
if 'pixels_per_unit' not in st.session_state:
    st.session_state['pixels_per_unit'] = None
if 'unit_label' not in st.session_state:
    st.session_state['unit_label'] = None

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
    pixel_distance = 1

    st.sidebar.subheader("📐 Scale Units")
    unit_options = ["microns", "nanometers", "millimeters", "inches"]
    selected_unit = st.sidebar.selectbox("Choose unit of measurement", unit_options, index=0)

    st.markdown("## ⚙️ Workflow Options")
    scale_toggle = st.toggle("Enable Scale Measurement", value=True)
    if not scale_toggle:
        st.info("⚠️ **Scale measurement disabled:** All fiber length measurements will be in pixels.")
    crop_toggle = st.toggle("Enable Cropping", value=True)

    # Logic flags
    skip_scale = not scale_toggle
    skip_crop = not crop_toggle

    # Fallback if scale is skipped
    if skip_scale:
        st.session_state["pixels_per_unit"] = 1.0
        st.session_state["unit_label"] = "pixels"

    # Sidebar Reset button at the end
    if st.sidebar.button("🔄 Reset Entire Workflow"):
        for key in ['uploaded_file', 'cropped_img', 'points', 'pixels_per_unit', 'unit_label']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # --- SCALE MEASUREMENT ---
    if scale_toggle:
        st.subheader("📏 Step 1: Measure Scale Bar (click two ends)")
        
        coords = streamlit_image_coordinates(pil_resized)

        if coords and len(st.session_state.points) < 2:
            scaled_point = (coords['x'] * scale_ratio, coords['y'] * scale_ratio)
            st.session_state.points.append(scaled_point)

        annotated = pil_original.copy()
        draw = ImageDraw.Draw(annotated)

        if len(st.session_state.points) >= 1:
            x1, y1 = st.session_state.points[0]
            draw.ellipse([(x1 - 5, y1 - 5), (x1 + 5, y1 + 5)], fill='red')
            draw.text((x1 + 6, y1), "A", fill='red')

        if len(st.session_state.points) == 2:
            x2, y2 = st.session_state.points[1]
            draw.ellipse([(x2 - 5, y2 - 5), (x2 + 5, y2 + 5)], fill='blue')
            draw.text((x2 + 6, y2), "B", fill='blue')
            draw.line([x1, y1, x2, y2], fill='yellow', width=2)

        # Info below the images
        if len(st.session_state.points) == 1:
            st.write(f"🔴 Point A: ({x1:.1f}, {y1:.1f})")

        elif len(st.session_state.points) == 2:
            pixel_distance = np.linalg.norm([x2 - x1, y2 - y1])
            st.write(f"🔴🔵 Points: A({x1:.1f}, {y1:.1f}) → B({x2:.1f}, {y2:.1f}) | Distance: `{pixel_distance:.2f}` pixels")

        col_a, col_b,_ = st.columns(3)
        with col_a:
            real_length = st.number_input(
                f"Enter real-world length of this line (in {selected_unit})", min_value=0.0001
            )

            if real_length:
                pixels_per_unit = pixel_distance / real_length
                st.session_state["pixels_per_unit"] = pixels_per_unit
                st.session_state["unit_label"] = selected_unit
                if real_length > 0.5:            
                    with col_b:
                        st.write(f"Pixel Conversion: {pixels_per_unit} pixels / {selected_unit}")

        if len(st.session_state.points) == 2 and st.button("🔁 Reset Measurement"):
            st.session_state.points = []
            st.session_state['pixels_per_unit'] = None

    # --- CROPPING ---
    if crop_toggle:
        st.subheader("🖼️ Crop Region of Interest")
        rect = st_cropper(pil_original, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
        cropped_img = np.array(rect)

        if st.button("📸 Confirm Crop"):
            st.session_state.cropped_img = cropped_img
            st.success("Crop Confirmed!")
            st.rerun()
    elif skip_crop:
        st.info("✂️ **Cropping skipped:** Full image will be used.")
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

    if not scale_toggle and not crop_toggle:
        st.info("ℹ️ **Scale measurement and cropping skipped:** Proceeding directly to preprocessing.")


if uploaded_file and (crop_toggle or skip_crop):
    st.subheader("🔍 Skeletonization Preview")
    def create_analyzer(image):
        return fiberL(
            image=image.copy(),
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

    # Usage in your app for preview:
    analyzer = create_analyzer(cropped_img)
    analyzer.preproc()

    col1, col2 = st.columns(2)
    with col1:
        st.image(cropped_img, caption="Cropped Original Image", use_container_width=True)
    with col2:
        st.image(255 - analyzer.sk_image * 255, caption="Skeletonized Image", use_container_width=True, clamp=True)

if st.session_state.get('uploaded_file') is not None:
    if skip_scale and skip_crop:
        st.warning("⚠️ You’ve disabled both scale and cropping. Default pixel units will be used on the **entire image**.")
    if st.button("🚀 Run Fiber Analysis"):
        if scale_toggle and st.session_state.get('pixels_per_unit') is None:
            st.error("⚠️ Scale measurement is enabled but not completed. Please click two points on the scale bar.")
        else:
            with st.spinner("Processing image... This may take a moment..."):
                analyzer.find_length()
                st.success("✅ Analysis Complete!")

                st.subheader("📊 Visual Summary")
                st.pyplot(analyzer.fig, use_container_width=True)

                st.subheader("📊 Summary Statistics")
                st.dataframe(analyzer.stats_df.style.format({"Value": "{:.2f}"}))

                with st.expander("📤 Export Results"):
                    analyzer.export_results_streamlit()
                    st.success("Results exported to fiberL_output folder!")
