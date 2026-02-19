import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm

# --- Page Config ---
st.set_page_config(page_title="HealthGuard AI", page_icon="🛡️")

@st.cache_resource
def load_my_model():
    # Update the path to match the root-relative location on Streamlit Cloud
    model_path = 'Brain_APP/healthguard_xception.keras'
    return tf.keras.models.load_model(model_path)

model = load_my_model()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return superimposed_img

st.title("🛡️ HealthGuard: Brain MRI Analysis")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocessing (224x224 as per your brain.py)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Start Analysis"):
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original MRI", use_column_width=True)
            if class_idx == 1:
                st.error("Result: Tumor Detected")
            else:
                st.success("Result: No Tumor Detected")

        # --- Grad-CAM Logic ---
        # For Xception, the last conv layer is typically 'block14_sepconv2_act'
        last_conv_layer = "block14_sepconv2_act" 
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        gradcam_img = save_and_display_gradcam(np.array(img_resized) * 255, heatmap)

        with col2:
            st.image(gradcam_img, caption="Explainable AI (Grad-CAM)", use_column_width=True)
            st.info("The heatmap highlights regions the AI used to determine the classification.")
