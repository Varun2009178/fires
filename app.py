
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import pickle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model



IMAGE_ADDRESS="https://insideclimatenews.org/wp-content/uploads/2023/03/wildfire_thibaud-mortiz-afp-getty-2048x1365.jpg"
IMAGE_SIZE=(224, 224)
IMAGE_NAME="user_image.png"
CLASS_LABEL=["nowildfire", "wildfire"]
CLASS_LABEL.sort()


@st.cache_resource
def get_MobileNetV2_model():
    base_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3), classes=1000, classifier_activation="softmax")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input, outputs=x)
    return model_frozen



@st.cache_resource
def load_sklearn_models(model_path):
    with open(model_path, "rb") as model_file:
        final_model = pickle.load(model_file)

    return final_model



def featurization(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)
    return predictions






MobileNetV2_featurize_model = get_MobileNetV2_model()
classification_model = load_sklearn_models("best_ml_model")


def run_app():
    st.title("Wildfire Prediction")
    st.subheader("Predict whether there is a chance of your satellite image's forest to have a wildfire")
    st.markdown("We use a comprehensive model with a 97 percent accuracy to ensure that your photo is correctly predicted to have a chance of a wildfire or not, completely for free.")
    st.image(IMAGE_ADDRESS, caption="Wildfire Classification Satellite Images")
    st.subheader("Please upload your satellite image")
    image = st.file_uploader("Please upload your satellite image", type=["jpg", "png", "jpeg"], accept_multiple_files=False, help="Upload an Image")
    if image:
        user_image = Image.open(image)
        user_image.save(IMAGE_NAME)
        st.image(user_image, caption="User uploaded an image")
        with st.spinner("Processing..."):
            image_features=featurization(IMAGE_NAME, MobileNetV2_featurize_model) 
            model_predict=classification_model.predict(image_features)
            result_label=CLASS_LABEL[model_predict[0]]
            st.success(f"Prediction: {result_label}")


    
run_app()
