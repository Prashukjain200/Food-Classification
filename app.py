import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# Constants
BINARY_CLASS_NAMES = ["Pizza", "Steak"]
MULTI_CLASS_NAMES = ["Chicken Curry", "Chicken Wings", "Fried Rice", "Grilled Salmon", "Hamburger", "Ice Cream",
                     "Pizza", "Ramen", "Steak", "Sushi"]

# Page configuration
st.set_page_config(layout="wide", page_title="Gourmet Vision", page_icon="🍽️")

# Custom CSS
st.markdown("""
<style>
    body {
        color: #fff;
        background-color: #0E1117;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 80rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF7676;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-top: 20px;
        border: 2px solid #FF4B4B;
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        box-shadow: 0 8px 15px rgba(255, 75, 75, 0.3);
        transform: translateY(-5px);
    }
    .food-type-list {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        border: 2px solid #FF4B4B;
    }
    .stSidebar {
        background-color: rgba(14, 17, 23, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
@st.cache_data
def load_and_prep_image(image, img_shape=224):
    img = Image.open(image).convert('RGB')
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(f'Model/{model_name}.h5')


def predict(model, image, class_names):
    pred = model.predict(image)
    if len(class_names) == 2:
        pred_class = class_names[int(tf.round(pred)[0][0])]
        confidence = pred[0][0] if pred_class == class_names[1] else 1 - pred[0][0]
    else:
        pred_class_index = np.argmax(pred, axis=1)[0]
        pred_class = class_names[pred_class_index]
        confidence = pred[0][pred_class_index]
    return pred_class, confidence


def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(255, 75, 75, 0.8)"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [50, 80], 'color': 'rgba(255, 255, 255, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(255, 255, 255, 0.5)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig


# Sidebar
with st.sidebar:
    st.title("🍽️ Gourmet Vision")
    st.image("https://img.icons8.com/fluent/96/000000/chef-hat.png", width=100)

    dataset = st.selectbox("Choose the dataset:",
                           ["Pizza/Steak (Binary)", "10 Food Types (Multi-class)"])

    model_choices = ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"] if dataset == "Pizza/Steak (Binary)" else [
        "Model 6", "Model 7", "Model 8"]
    model_choice = st.selectbox("Choose the model:", model_choices)

    uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Main content
st.title("🍽️ Gourmet Vision: AI Food Classifier")
st.write("Upload an image and let our AI chef identify your dish!")

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            with st.spinner("Our AI chef is analyzing your dish..."):
                # Simulate a delay for dramatic effect
                time.sleep(2)

                # Prepare image
                img = load_and_prep_image(uploaded_file)

                # Load model and predict
                model = load_model(model_choice)
                pred_class, confidence = predict(model, img,
                                                 BINARY_CLASS_NAMES if dataset == "Pizza/Steak (Binary)" else MULTI_CLASS_NAMES)

                # Display prediction with animation
                st.balloons()
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Bon Appétit! 👨‍🍳</h2>
                    <p>Our AI chef is {confidence * 100:.2f}% confident that your dish is <strong>{pred_class}</strong>.</p>
                </div>
                """, unsafe_allow_html=True)

                # Display confidence gauge
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)

with col2:
    st.markdown("""
    <div class="food-type-list">
        <h3>🍽️ Menu</h3>
    """, unsafe_allow_html=True)

    if dataset == "Pizza/Steak (Binary)":
        for food in BINARY_CLASS_NAMES:
            st.markdown(f"- 🍽️ {food}")
    elif dataset == "10 Food Types (Multi-class)":
        for food in MULTI_CLASS_NAMES:
            st.markdown(f"- 🍽️ {food}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Add a fun fact section
    st.markdown("""
    <div class="food-type-list" style="margin-top: 20px;">
        <h3>🍳 Chef's Fun Fact</h3>
        <p>Did you know? The word 'restaurant' comes from the French word 'restaurer', which means 'to restore'!</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr>
<p style='text-align: center;'>Crafted with 🧡 by Prashuk | 
<a href='https://github.com/Prashukjain200' target='_blank'>GitHub</a> | 
<a href='https://prashuk.netlify.app/' target='_blank'>Portfolio</a></p>
""", unsafe_allow_html=True)
