import streamlit as st
import pickle
import numpy as np
from PIL import Image
import requests
import os
import io # Kept io, though not strictly needed here

# --- Conditional Imports (Required for Deployment Logic) ---
try:
    from tensorflow.keras.models import load_model
except ImportError:
    pass

# ==============================================================================
# 0. CONFIGURATION AND FILE PATHS
# ==============================================================================

# --- Model Parameters and Paths ---
CROP_MODEL_PATH = "model_cache/model.pkl"
DISEASE_MODEL_PATH = "model_cache/plant_disease_prediction_model.h5"
IMAGE_SIZE = (224, 224) # Corrected size from your notebook

# --- Label Mappings ---
crop_label_mapping = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

disease_label_mapping = {
    0: 'Healthy',
    1: 'Apple Scab',
    2: 'Potato Early Blight',
    3: 'Tomato Mosaic Virus',
    # ADD ALL YOUR DISEASE CLASSES HERE
}


# ==============================================================================
# 1. CORE DOWNLOAD AND LOADING FUNCTIONS (MODIFIED FOR GOOGLE DRIVE)
# ==============================================================================

def download_file(url, filename):
    """Downloads a file from a URL to a local path (used for GDrive hosting)."""
    st.info(f"Downloading {os.path.basename(filename)}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() 
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Successfully downloaded {os.path.basename(filename)}.")
    except Exception as e:
        st.error(f"❌ Failed to download {os.path.basename(filename)}. Check URL/Secrets/Permissions: {e}")
        st.stop() 


@st.cache_resource
def load_crop_model():
    """Loads the crop recommendation model (model.pkl)."""
    model_url = st.secrets.get("CROP_MODEL_URL")
    if not os.path.exists(CROP_MODEL_PATH):
        if not model_url:
            st.warning("Crop model not found locally. Add CROP_MODEL_URL to secrets for cloud deployment.")
            return None # Skip download if running locally without secrets
        download_file(model_url, CROP_MODEL_PATH)
        
    try:
        return pickle.load(open(CROP_MODEL_PATH, "rb")) 
    except Exception as e:
        st.error(f"❌ Error loading crop model (Check scikit-learn==1.6.1 version!): {e}")
        return None

@st.cache_resource
def load_disease_model():
    """Loads the plant disease prediction model (plant_disease_prediction_model.h5)."""
    model_url = st.secrets.get("DISEASE_MODEL_URL")

    if not os.path.exists(DISEASE_MODEL_PATH):
        if not model_url:
            st.warning("Disease model not found locally. Add DISEASE_MODEL_URL to secrets for cloud deployment.")
            return None
        download_file(model_url, DISEASE_MODEL_PATH)
        
    try:
        # load_model is imported conditionally, so we call it here
        from tensorflow.keras.models import load_model 
        return load_model(DISEASE_MODEL_PATH) 
    except Exception as e:
        st.error(f"❌ Error loading disease model (Check TensorFlow install): {e}")
        return None

# --- Load Models (The app starts downloading if paths don't exist) ---
crop_model = load_crop_model()
disease_model = load_disease_model()


# ==============================================================================
# 2. Image Preprocessing Function
# ==============================================================================

def preprocess_image(uploaded_file):
    """Resizes, converts, and normalizes the uploaded image for CNN input."""
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==============================================================================
# 3. Streamlit App Layout
# ==============================================================================

st.set_page_config(
    page_title="Kisan Sathi", # Updated Title
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Kisan Sathi: The Farmer's AI Companion") # Updated Title
st.markdown("---")

# --- SIDEBAR NAVIGATION (Cosmetic Changes) ---
st.sidebar.header("🌿 How may I assist you?")
tool_choice = st.sidebar.radio(
    "Select a service:", # Simple sentence
    ("Crop Recommendation", "Plant Disease Prediction")
)

# ----------------------------------------------
# A. Crop Recommendation Section (UI IMPROVEMENTS)
# ----------------------------------------------
if tool_choice == "Crop Recommendation":
    st.header("🌾 Best Crop Recommendation")
    st.markdown("Use the sliders to input soil nutrients and climate parameters to find the optimal crop.")

    if crop_model:
        # Use a form to group inputs and prevent constant re-running
        with st.form("crop_form"):
            col1, col2, col3 = st.columns(3)

            # NPK inputs (Core Soil Health)
            with col1:
                st.subheader("Nutrients (PPM)")
                # Using Sliders for better UX
                nitrogen = st.slider("Nitrogen (N)", min_value=0, max_value=140, value=90, help="Nitrogen concentration in the soil.")
                phosphorus = st.slider("Phosphorus (P)", min_value=0, max_value=145, value=42, help="Phosphorus concentration in the soil.")
                potassium = st.slider("Potassium (K)", min_value=0, max_value=210, value=40, help="Potassium concentration in the soil.")

            # Climate Inputs
            with col2:
                st.subheader("Climate")
                temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=45.0, value=25.0, step=0.1)
                humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
                rainfall = st.slider("Rainfall (mm)", min_value=20.0, max_value=300.0, value=200.0, step=0.1)
            
            # pH (Soil Acidity)
            with col3:
                st.subheader("Soil pH")
                ph = st.slider("pH Value", min_value=3.0, max_value=10.0, value=6.5, step=0.1, help="Soil acidity/alkalinity level.")
            
            st.markdown("---")
            submit_button = st.form_submit_button(label="Predict Optimal Crop 🔎")

        if submit_button:
            # Use st.status for clear progress bar
            with st.status('Analyzing conditions and predicting...', expanded=True) as status:
                try:
                    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

                    # Make prediction
                    prediction = crop_model.predict(features)[0]
                    crop_name = crop_label_mapping.get(int(prediction), "Unknown")

                    status.update(label=f"✅ Prediction Complete! Optimal Crop Found.", state="complete")
                    
                    # Display result prominently
                    st.markdown("### Prediction Result")
                    st.metric(label="Optimal Crop to Grow", value=crop_name.capitalize(), delta="High Confidence")
                    st.balloons()

                except Exception as e:
                    status.update(label=f"❌ Prediction Failed", state="error")
                    st.error(f"Prediction Error: {e}. Check your model and inputs.")
    else:
        st.warning("Cannot run Crop Recommendation. Model failed to load.")

# ----------------------------------------------
# B. Plant Disease Prediction Section (UI IMPROVEMENTS)
# ----------------------------------------------
elif tool_choice == "Plant Disease Prediction":
    st.header("🍄 Plant Disease Prediction from Image")
    st.markdown("Upload a close-up image of a leaf for instant diagnosis.")

    if disease_model:
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            
            with col_info:
                if st.button("Predict Plant Condition 🔍", use_container_width=True):
                    with st.status('Analyzing image...', expanded=True) as status:
                        try:
                            # 1. Preprocess the image
                            processed_image = preprocess_image(uploaded_file)

                            # 2. Make prediction
                            predictions = disease_model.predict(processed_image)
                            predicted_class_index = np.argmax(predictions, axis=1)[0]
                            confidence = np.max(predictions) * 100
                            disease_name = disease_label_mapping.get(predicted_class_index, "Unknown Disease")

                            status.update(label="✅ Analysis Complete!", state="complete")

                            st.markdown("### Prediction Result")
                            if 'Healthy' in disease_name:
                                st.success(f"The plant is predicted to be **{disease_name}**! 🌿")
                                st.balloons()
                            else:
                                st.error(f"The predicted plant condition is: **{disease_name}** 🔴")
                            
                            st.info(f"Confidence: {confidence:.2f}%")

                        except Exception as e:
                            status.update(label=f"❌ Analysis Failed", state="error")
                            st.error(f"Prediction Error: {e}. Check if the image format is compatible.")
    else:
        st.warning("Cannot run Disease Prediction. Model failed to load.")

# End of app.py
