import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
from preprocessing_images import *
from tensorflow.keras.models import load_model

st.title('Brain MRI segmentation application')

st.markdown("***")

st.subheader('Upload the MRI scan of the brain')
option = st.radio('', ('Single MRI scan', 'Multiple MRI scans'))
st.write('You selected:', option)

model = load_model('model_UNet.h5')  # Load your trained model

def predict_tumor(img):
    # Assuming your preprocessing function returns a numpy array suitable for prediction
    processed_img = load_preprocess_image(str(img))
    prediction = model.predict(np.expand_dims(processed_img, axis=0))
    return prediction

if option == 'Single MRI scan':
    st.subheader('Upload the MRI scan of the brain')
    uploaded_file = st.file_uploader(' ', accept_multiple_files=False)

    if uploaded_file is not None:
        img = final_fun_1(uploaded_file)
        st.image(img)
        prediction = predict_tumor(img)
        st.write(f"Prediction: {prediction}")

    else:
        st.write("Make sure your image is in TIF/JPG/PNG format.")

elif option == 'Multiple MRI scans':
    st.subheader('Upload the MRI scans of the brain')
    uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
    if uploaded_files:
        st.write("Images Uploaded Successfully")
        for uploaded_file in uploaded_files:
            img = final_fun_1(uploaded_file)
            st.image(img)
            prediction = predict_tumor(img)
            st.write(f"Prediction: {prediction}")

    else:
        st.write("Make sure your images are in TIF/JPG/PNG format.")

st.markdown("***")

result = st.button('Try again')
if result:
    st.experimental_rerun()
