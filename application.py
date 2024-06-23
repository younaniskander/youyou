from interface_tumor import *
import streamlit as st
from utils import init_session_state_variables, dataset_unzip, rename_wrong_file, check_if_dataset_exists
from variables import data_path
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

def init_app():
    """
    App Configuration
    This function sets & displays the app title, its favicon, and initializes some session_state values.
    It also verifies that the dataset exists in the environment and is well unzipped.
    """

    # Set config and app title
    st.set_page_config(page_title="Image Segmentation", layout="wide", page_icon="🧠")
    st.title("Brain Tumors Segmentation 🧠")

    # Initialize session state variables
    init_session_state_variables()

    # Unzip dataset if not already done
    dataset_unzip()

    # Rename the 355th file if necessary (it has a default incorrect name)
    rename_wrong_file(data_path)

    # Check if the dataset exists in the environment to know if we can launch the app
    check_if_dataset_exists()

    # Load the pre-trained model
    model = load_model('model_UNet.h5')

    return model

def load_preprocess_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image / 256.0
    return image

def predict_tumor(model, image_path):
    """Reads a brain MRI image and returns the prediction."""
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    X = np.empty((1, 256, 256, 3))
    X[0,] = img
    prediction = model.predict(X)
    return prediction

def launch_app(model):
    st.subheader('Upload the MRI scan of the brain')
    option = st.radio('', ('Single MRI scan', 'Multiple MRI scans'))
    st.write('You selected:', option)

    if option == 'Single MRI scan':
        uploaded_file = st.file_uploader(' ', accept_multiple_files=False)

        if uploaded_file is not None:
            st.image(uploaded_file)
            prediction = predict_tumor(model, uploaded_file)
            st.write(f"Prediction: {prediction}")

        else:
            st.write("Make sure your image is in TIF/JPG/PNG format.")

    elif option == 'Multiple MRI scans':
        uploaded_files = st.file_uploader(' ', accept_multiple_files=True)
        if uploaded_files:
            st.write("Images Uploaded Successfully")
            for uploaded_file in uploaded_files:
                st.image(uploaded_file)
                prediction = predict_tumor(model, uploaded_file)
                st.write(f"Prediction: {prediction}")

        else:
            st.write("Make sure your images are in TIF/JPG/PNG format.")

    st.markdown("***")

    result = st.button('Try again')
    if result:
        st.experimental_rerun()

if __name__ == '__main__':
    model = init_app()
    launch_app(model)
