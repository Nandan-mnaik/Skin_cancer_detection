import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Model loading
@st.cache_resource
def load_classification_model():
    return load_model('models/ham10000_vitmodel.keras')

model = load_classification_model()

# Different types of cancer with their descriptions
cancer_types = {
    'Actinic keratoses': "Actinic keratoses are dry, scaly patches of skin that have been damaged by long-term exposure to ultraviolet (UV) radiation, typically from the sun. These lesions are considered precancerous and may develop into skin cancer if left untreated.",
    'Basal cell carcinoma': "Basal cell carcinoma is the most common type of skin cancer. It typically develops on areas of skin exposed to the sun, such as the face and neck. While it rarely spreads to other parts of the body, it can grow and invade nearby tissues if left untreated.",
    'Benign keratosis-like lesions': "Benign keratosis-like lesions are non-cancerous growths on the skin. They are often harmless and don't require treatment unless they cause discomfort or for cosmetic reasons. These lesions can sometimes be mistaken for more serious conditions, so professional evaluation is important.",
    'Dermatofibroma': "Dermatofibroma is a common, benign skin tumor that most often appears on the legs, but may occur anywhere on the body. It's usually small, firm, and painless, often developing after a minor injury to the skin.",
    'Melanoma': "Melanoma is the most serious type of skin cancer. It develops in the cells that produce melanin, the pigment that gives skin its color. Melanoma can occur anywhere on the body, even in areas not typically exposed to the sun. Early detection and treatment are crucial for the best outcomes.",
    'Melanocytic nevi': "Melanocytic nevi, commonly known as moles, are usually benign growths on the skin. They occur when pigment-producing cells in the skin grow in clusters or clumps. While most moles are harmless, some can develop into melanoma over time, so it's important to monitor them for changes.",
    'Vascular lesions': "Vascular lesions are relatively common abnormalities of the skin and underlying tissues, affecting the blood vessels. They can be present at birth (congenital) or develop later in life. While many are benign, some types may require treatment for medical or cosmetic reasons."
}

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    return {cancer: float(prob) for cancer, prob in zip(cancer_types.keys(), prediction)}

st.title('Skin Cancer Classification App')

st.write("""
This app uses a MobileNetV3Large model trained on the HAM10000 dataset to classify skin lesions.
Upload an image to get a prediction and detailed information about the predicted condition.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    predictions = predict(image)
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_prediction, confidence = sorted_predictions[0]
    
    st.subheader("Analysis Results")
    st.write(f"Based on the uploaded image, our model predicts that this lesion is most likely **{top_prediction}** with a confidence of **{confidence:.2f}**.")
    
    st.write(f"**About {top_prediction}:**")
    st.write(cancer_types[top_prediction])
    
    st.subheader("Probability Distribution")
    for cancer, prob in sorted_predictions:
        st.write(f"{cancer}: {prob:.2%}")
        st.progress(prob)
    
    st.subheader("Important Notes")
    st.warning("This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for proper evaluation of any skin concerns.")

    if confidence < 0.7:
        st.info("The confidence level for this prediction is relatively low. It's particularly important to seek professional medical advice for a proper diagnosis.")

    st.write("Remember to perform regular self-examinations of your skin and consult a dermatologist if you notice any unusual changes or growths.")