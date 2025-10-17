import streamlit as st
import pickle
import numpy as np
from PIL import Image
import io

# TensorFlow/Keras is required for the disease model
try:
    from tensorflow.keras.models import load_model
except ImportError:
    # This will be caught later in the app with a warning
    pass


# ==============================================================================
# 1. Model and Data Loading
# ==============================================================================

@st.cache_resource
def load_crop_model():
    """Loads the crop recommendation model (model.pkl)."""
    try:
        return pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.error("❌ Error: 'model.pkl' not found. Ensure the crop model file is present.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading crop model: {e}")
        return None


@st.cache_resource
def load_disease_model():
    """Loads the plant disease prediction model (plant_disease_prediction_model.h5)."""
    try:
        # NOTE: Using the specified filename
        return load_model("plant_disease_prediction_model.h5")
    except FileNotFoundError:
        st.error("❌ Error: 'plant_disease_prediction_model.h5' not found. Ensure the disease model file is present.")
        return None
    except NameError:
        st.error("❌ Error: TensorFlow/Keras module missing. Install it with: pip install tensorflow")
        return None
    except Exception as e:
        st.error(f"❌ Error loading disease model: {e}")
        return None


crop_model = load_crop_model()
disease_model = load_disease_model()

# Label mapping for Crop Model (Same as your initial setup)
crop_label_mapping = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# Label mapping for Disease Model (You MUST customize this based on your model's output)
disease_label_mapping = {
    0: 'Healthy',
    1: 'Apple Scab',
    2: 'Potato Early Blight',
    3: 'Tomato Mosaic Virus',
    # ADD ALL YOUR DISEASE CLASSES HERE
}

# Model expected image size (Common CNN input size)
IMAGE_SIZE = (224,224)


# ==============================================================================
# 2. Image Preprocessing Function
# ==============================================================================

def preprocess_image(uploaded_file):
    """Resizes, converts, and normalizes the uploaded image for CNN input."""
    img = Image.open(uploaded_file).convert('RGB')  # Ensure 3 channels
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)

    # Normalize (0-255 -> 0-1)
    img_array = img_array / 255.0

    # Expand dimensions (for batch size 1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==============================================================================
# 3. Streamlit App Layout
# ==============================================================================

st.set_page_config(
    page_title="Kisan Saathi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧑‍🌾 The Farmer's Digital Companion")
st.markdown(
    "Use the sidebar to choose between **Crop Recommendation** (based on soil/climate) and **Plant Disease Prediction** (based on image).")

# Create a sidebar for navigation
st.sidebar.header("🌿 How may I assist you?")
tool_choice = st.sidebar.radio(
    "Select a service.:",
    ("Crop Recommendation", "Plant Disease Prediction")
)

# ----------------------------------------------
# A. Crop Recommendation Section
# ----------------------------------------------
if tool_choice == "Crop Recommendation":
    st.header("🌱 Best Crop Recommendation")

    if crop_model:
        st.info("Input the features below and click 'Predict' for the optimal crop.")

        # Input fields
        col1, col2 = st.columns(2)

        with col1:
            nitrogen = st.number_input("Nitrogen (N) - (e.g., 90)", min_value=0.0, step=1.0)
            potassium = st.number_input("Potassium (K) - (e.g., 40)", min_value=0.0, step=1.0)
            temperature = st.number_input("Temperature (°C) - (e.g., 25.0)", min_value=0.0, step=0.1)
            ph = st.number_input("pH Value - (e.g., 6.5)", min_value=0.0, max_value=14.0, step=0.1)

        with col2:
            phosphorus = st.number_input("Phosphorus (P) - (e.g., 42)", min_value=0.0, step=1.0)
            humidity = st.number_input("Humidity (%) - (e.g., 80.0)", min_value=0.0, max_value=100.0, step=0.1)
            rainfall = st.number_input("Rainfall (mm) - (e.g., 200.0)", min_value=0.0, step=0.1)

        # Prediction button
        if st.button("Predict Optimal Crop"):
            with st.spinner('Predicting...'):
                try:
                    features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

                    # Make prediction
                    prediction = crop_model.predict(features)[0]
                    crop_name = crop_label_mapping.get(int(prediction), "Unknown")

                    # Display result
                    st.success("### ✅ Prediction Result")
                    st.balloons()
                    st.markdown(f"The best crop to be cultivated is **{crop_name.capitalize()}** 🌱")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")
    else:
        st.warning("Cannot run Crop Recommendation. Model failed to load.")

# ----------------------------------------------
# B. Plant Disease Prediction Section
# ----------------------------------------------
elif tool_choice == "Plant Disease Prediction":
    st.header("🍄 Plant Disease Prediction from Image")

    if disease_model:
        st.info("Upload a close-up image of a leaf. The predicted disease or health status will appear below.")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            if st.button("Predict Plant Condition"):
                with st.spinner('Analyzing image...'):
                    try:
                        # 1. Preprocess the image
                        processed_image = preprocess_image(uploaded_file)

                        # 2. Make prediction
                        predictions = disease_model.predict(processed_image)
                        predicted_class_index = np.argmax(predictions, axis=1)[0]
                        confidence = np.max(predictions) * 100

                        # 3. Get disease name
                        disease_name = disease_label_mapping.get(predicted_class_index, "Unknown Disease")

                        # 4. Display result
                        st.success("### ✅ Prediction Result")
                        if 'Healthy' in disease_name:
                            st.markdown(f"The plant is predicted to be **{disease_name}**! 🌿")
                        else:
                            st.error(f"The predicted plant condition is: **{disease_name}** 🔴")
                        st.info(f"Confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
    else:
        st.warning("Cannot run Disease Prediction. Model failed to load.")

# End of app.py