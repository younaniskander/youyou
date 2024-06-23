import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io

# Dummy user database
users = {
    "admin": "password123",
    "user": "testpass"
}

st.title('Brain MRI Segmentation Application')

st.markdown("***")

st.subheader('Upload the MRI scan of the brain')

def preprocess_image(image, IMG_SIZE=(192, 192)):
    """
    Preprocess the image: resize and normalize.
    Ensure that the processed image has 3 color channels (RGB).
    """
    # Resize the image to match the input shape expected by the model
    image = image.resize(IMG_SIZE)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Ensure RGB format (remove alpha channel if present)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values to be between 0 and 1
    normalized_image = image_array / 255.0

    # Expand dimensions to match model input shape
    processed_image = np.expand_dims(normalized_image, axis=0)
    
    return processed_image

def model_page():
    st.title("Lung Cancer Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(image)
        st.image(processed_image[0], caption='Processed Image', use_column_width=True)
        model_path = 'model_Unet.h5'
        model = tf.keras.models.load_model(model_path)
        try:
            prediction = model.predict(processed_image)
            classes = ['normal', 'adenocarcinoma', 'large.cell', 'squamous']
            predicted_class = classes[np.argmax(prediction)]
            st.write('Prediction:', predicted_class)
        except Exception as e:
            st.error(f"Error predicting: {e}")

def main():
    model_page()

if __name__ == "__main__":
    main()
