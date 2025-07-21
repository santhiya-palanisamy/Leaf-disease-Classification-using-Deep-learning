import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the model without the optimizer
loaded_model = load_model('trained_plant_disease_model.keras', compile=False)

# Compile the model with a new optimizer (e.g., Adam with a specific learning rate)
loaded_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Tensorflow Model Prediction
def model_prediction(test_image):
    #loaded_model = load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = loaded_model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("LEAF DISEASE CLASSIFICATION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Leaf Disease Classification System! 🌿🔍

     **Especially For Tomato Plants**

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Team mates
                1. Santhiya P - Project leader
                2. Nandhana Devi C - Dataset Collection
                3. Swathi T - Tester
                #### About Dataset
                This dataset is consist of Healthy and Diseased leaves of Tomato Plants.
                11 classes of sets and total images 5224.
                #### Content
                1. train (5224 images)
                2. validation (5224 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Background_without_leaves','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        st.success("Model is predicting it's a {}".format(class_name[result_index]))
