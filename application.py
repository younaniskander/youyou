import streamlit as st
import numpy as np
import pandas as pd
import time
from streamlit import caching
from PIL import Image
from preprocessing_images import *
from keras.models import load_model
import tensorflow as tf

# Load your model
@st.cache_resource
def load_my_model():
    model = load_model('model_UNet.h5')
    return model

model = load_my_model()

st.title('Brain MRI Segmentation Application')

st.markdown("***")

st.subheader('Upload the MRI scan of the brain')
option = st.radio('', ('Single MRI scan', 'Multiple MRI scans'))
st.write('You selected:', option)

def preprocess_image(image):
    # Convert the image to the format expected by the model
    image = image.resize((224, 224))  # Assuming the model expects 224x224 input images
    image = np.array(image)
    if len(image.shape) == 2:  # if grayscale, convert to RGB
        image = np.stack((image,)*3, axis=-1)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

if option == 'Single MRI scan':
    st.subheader('Upload the MRI scan of the brain')
    uploaded_file = st.file_uploader(' ', accept_multiple_files=False)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

        st.write("Classifying...")
        prediction = predict(image, model)
        st.write("Prediction: ", prediction)
        
    else:
        st.write("Make sure your image is in TIF/JPG/PNG Format.")

elif option == 'Multiple MRI scans':
    st.subheader('Upload the MRI scans of the brain')
    uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
    if len(uploaded_files) != 0:
        st.write("Images Uploaded Successfully")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

            st.write("Classifying...")
            prediction = predict(image, model)
            st.write("Prediction: ", prediction)
            
    else:
        st.write("Make sure your images are in TIF/JPG/PNG Format.")

st.markdown("***")

result = st.button('Try again')
if result:
    st.experimental_rerun()
