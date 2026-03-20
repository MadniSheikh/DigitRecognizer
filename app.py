import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import pickle
import base64
from io import BytesIO

# --- Set Page Config ---
st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

# --- 1. Load Assets ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("./model/ann_model1.h5")
    with open("./utils/scaling.pkl", 'rb') as file:
        scaling = pickle.load(file)
    return model, scaling

model, scaler = load_assets()

# --- 2. Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #1e1e1e;
    }
    
    /* STRICT CANVAS FIX */
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        width: 300px !important;
        height: 300px !important;
        border: none !important;
        outline: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        border-radius: 8px !important;
        margin: 0 auto !important;
        display: block !important;
        background-color: black !important;
    }

    button[title="View fullscreen"] { display: none !important; }

    /* CLEAR BUTTON STYLING */
    div.stButton > button[kind="secondary"] {
        background-color: #FFC107 !important; 
        color: #000000 !important; 
        border: 2px solid #E0A800 !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    /* PREDICT BUTTON STYLING */
    div.stButton > button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        border: none !important;
        width: 100% !important;
    }
    
    /* MATCHING THE SKETCH: Rectangular Results Card */
    .results-card {
        background-color: white;
        padding: 15px 20px; /* Reduced vertical padding forces the rectangle shape */
        border-radius: 8px;
        border: 2px solid #ccc; /* Adding the border from your sketch */
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 140px; 
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Simplified Session State ---
if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = 0

def clear_canvas():
    st.session_state['canvas_key'] += 1

# --- 4. Preprocessing Logic ---
def preprocess_image(img):
    img = img.filter(ImageFilter.MaxFilter(3)) 
    w, h = img.size
    if w > h:
        new_w, new_h = 20, max(1, int(20 * (h / w)))
    else:
        new_w, new_h = max(1, int(20 * (w / h))), 20
        
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    img_array = np.array(img_resized)

    total_mass = np.sum(img_array)
    if total_mass > 0:
        y_indices, x_indices = np.indices(img_array.shape)
        cy = np.sum(y_indices * img_array) / total_mass
        cx = np.sum(x_indices * img_array) / total_mass
    else:
        cy, cx = new_h / 2, new_w / 2

    canvas_bg = np.zeros((28, 28), dtype=np.uint8)
    y_offset = int(np.round(14.0 - cy))
    x_offset = int(np.round(14.0 - cx))

    y1, y2 = max(0, y_offset), min(28, y_offset + new_h)
    x1, x2 = max(0, x_offset), min(28, x_offset + new_w)
    sy1, sy2 = max(0, -y_offset), max(0, -y_offset) + (y2 - y1)
    sx1, sx2 = max(0, -x_offset), max(0, -x_offset) + (x2 - x1)

    canvas_bg[y1:y2, x1:x2] = img_array[sy1:sy2, sx1:sx2]
    return Image.fromarray(canvas_bg)

# --- 5. Streamlit UI Layout ---

st.markdown("<h1 style='text-align: center; font-size: 50px; font-weight: 900; color: #333;'>✍️ Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 22px; margin-bottom: 40px;'>Draw a digit (0-9) smoothly and click Predict.</p>", unsafe_allow_html=True)

col1, spacer, col2 = st.columns([1.2, 0.2, 1.2])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        display_toolbar=False, 
        key=f"canvas_{st.session_state['canvas_key']}",
    )

    st.button("🗑️ Clear Canvas", on_click=clear_canvas, use_container_width=True)
    predict_clicked = st.button("✨ Predict Digit", type="primary", use_container_width=True)

with col2:
    if predict_clicked:
        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data
            img = Image.fromarray(img_array.astype('uint8'), 'RGBA').convert('L')
            bbox = img.getbbox()
            
            if bbox is None:
                st.warning("Please draw a digit first!")
            else:
                img = img.crop(bbox)
                img = preprocess_image(img)
                img = img.filter(ImageFilter.SHARPEN)
                
                # Predict
                img_flat = np.array(img).reshape(1, 784)
                img_scaled = scaler.transform(img_flat)
                prediction = model.predict(img_scaled)
                digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Convert PIL image to base64 so we can put it in a strict HTML square box
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 1. "What model analyzed" + Square Image Box (Matches Sketch)
                st.markdown("<h4 style='text-align: center; color: #555; margin-bottom: 10px; font-family: sans-serif;'>What model analyzed</h4>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: center; margin-bottom: 25px;">
                        <div style="border: 2px solid #ccc; width: 150px; height: 150px; background-color: black; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                            <img src="data:image/png;base64,{img_str}" style="width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated;">
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 2. Rectangular Prediction Card (Matches Sketch)
                st.markdown(f"""
                <div class="results-card">
                    <h3 style="color: #666; margin-bottom: 5px; font-size: 20px;">Prediction: {digit}</h3>
                    <p style="color: #4CAF50; font-size: 18px; margin: 5px 0 0 0; font-weight: bold;">Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        # Empty state placeholder
        st.markdown("""
        <div class="results-card" style="opacity: 0.5; margin-top: 50px; border: 2px dashed #ccc;">
            <p style="margin: 30px 0; color: #888; font-size: 16px;">Draw a number and click Predict to see the results here.</p>
        </div>
        """, unsafe_allow_html=True)