import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained model
model = load_model('model_UNet.h5')

def load_preprocess_image(img):
    """Preprocess the uploaded image to the required format."""
    im = Image.open(img)
    image = np.array(im)
    image = cv2.resize(image, (256, 256))
    image = np.array(image, dtype=np.float64)
    image -= image.mean()
    image /= image.std()
    return image

def predict_tumor(model, img):
    """Predict the tumor using the pre-trained model."""
    processed_img = load_preprocess_image(img)
    X = np.expand_dims(processed_img, axis=0)
    prediction = model.predict(X)
    return prediction

st.title('Brain MRI Segmentation Application')

# Upload the MRI scan of the brain
st.subheader('Upload the MRI scan of the brain')
uploaded_file = st.file_uploader(' ', accept_multiple_files=False, type=['png', 'jpg', 'jpeg', 'tif'])

# Predict button
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI', use_column_width=True)
    
    if st.button('Predict'):
        prediction = predict_tumor(model, uploaded_file)
        st.write(f"Prediction: {prediction}")
else:
    st.write("Please upload an MRI scan in PNG/JPG/JPEG/TIF format.")
