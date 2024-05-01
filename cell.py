import streamlit as st
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Title of the web app
st.title("Malaria Disease Detection using Red Blood Cells")

# Displaying an initial image
img_path = r'D:\DL Projects\cell_images\images\malaria.jpeg'
img = Image.open(img_path)
st.image(img, width=500)

st.text("Upload the sample image to check if it's parasitized or uninfected.")

# Load pre-trained model
model = tf.keras.models.load_model("malaria_cells.h5")

def preprocess_image(image):
    """
    Function to preprocess the image for prediction.
    """
    image = Image.open(image)
    image = image.resize((100,100))
    image_array=tf.keras.preprocessing.image.img_to_array(image)
    image_array=image_array/255.0
    return image_array.reshape(1,100,100,3)

def predict_image(image_file):
    """
    Function to use the model to predict the image.
    """
    preprocessed_image=preprocess_image(image_file)
    predictions=model.predict(preprocessed_image)
    predictions=predictions.argmax(axis=1)

    if predictions==0:
        shots="The given sample is parasitized"
    else:
        shots="The given sample is uninfected"
    return shots    

# File uploader widget
uploaded_file=st.file_uploader('choose an image :',type=["jpg","jpeg","png","webp"])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded image",use_column_width=True)
    predicted_image=predict_image(uploaded_file)
    st.success(f"Result\:{predicted_image}")