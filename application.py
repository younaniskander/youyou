import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2
from tensorflow.keras.models import load_model

# Importing required functions and models from your preprocessing_images and efficient_Unet scripts
from efficient_Unet import build_effienet_unet

st.title('Brain MRI Segmentation Application')

st.markdown("***")

st.subheader('Upload the MRI scan of the brain')
option = st.radio('', ('Single MRI scan', 'Multiple MRI scans'))
st.write('You selected:', option)

# Load your trained models
input_shape = (256, 256, 3)
effienet_Unet_model = build_effienet_unet(input_shape)
effienet_Unet_model.load_weights("tf_effienet_Unet_brain_final")
model_UNet = load_model('model_UNet.h5')

def load_preprocess_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image / 256.0
    return image

def predict_segmentation_mask(image_path, model):
    """Reads a brain MRI image and returns the segmentation mask of the image."""
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    X = np.empty((1, 256, 256, 3))
    X[0,] = img
    predict = model.predict(X)
    return predict.reshape(256, 256, 3)

def plot_MRI_predicted_mask(original_img, predicted_mask):
    """Plots both the original image and predicted mask side by side."""
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    axes[0].imshow(original_img)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Original MRI')
    axes[1].imshow(predicted_mask)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Predicted Mask')
    fig.tight_layout()
    filename = 'pair' + str(random.randint(100, 1000)) + str(random.randint(100, 1000)) + '.png'
    plt.savefig(filename)
    return filename

def final_fun_1(image_path, model):
    '''Input: Image path through the upload method.
       Returns: combined image of original and predicted mask.
    '''
    image = load_preprocess_image(image_path)
    mask = predict_segmentation_mask(image_path, model)
    combined_img = plot_MRI_predicted_mask(original_img=image, predicted_mask=mask)
    return combined_img

if option == 'Single MRI scan':
    st.subheader('Upload the MRI scan of the brain')
    uploaded_file = st.file_uploader(' ', accept_multiple_files=False)

    if uploaded_file is not None:
        img = final_fun_1(uploaded_file, model_UNet)  # You can switch between model_UNet and effienet_Unet_model
        st.image(img)
    else:
        st.write("Make sure your image is in TIF/JPG/PNG format.")

elif option == 'Multiple MRI scans':
    st.subheader('Upload the MRI scans of the brain')
    uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
    if uploaded_files:
        st.write("Images Uploaded Successfully")
        for uploaded_file in uploaded_files:
            img = final_fun_1(uploaded_file, model_UNet)  # You can switch between model_UNet and effienet_Unet_model
            st.image(img)
    else:
        st.write("Make sure your images are in TIF/JPG/PNG format.")

st.markdown("***")

result = st.button('Try again')
if result:
    st.experimental_rerun()
