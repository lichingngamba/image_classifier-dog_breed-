# Import required libraries
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Create the ResNet50 model
model = ResNet50(weights='imagenet')

# Create a function to classify images
def classify_image(image_file):
    # Load the image using Keras image module
    img = image.load_img(image_file, target_size=(224, 224))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Reshape the image to add a dimension for batch size
    img_batch = np.expand_dims(img_array, axis=0)
    # Preprocess the image for the ResNet50 model
    img_preprocessed = preprocess_input(img_batch)
    # Use the ResNet50 model to predict the class probabilities
    prediction = model.predict(img_preprocessed)
    # Decode the predictions to get the class labels
    decoded_predictions = decode_predictions(prediction, top=3)[0]
    # Return the top 3 predicted labels and their probabilities
    return [(label, float(prob)*100) for (class_id, label, prob) in decoded_predictions]

# Create a Streamlit app
def app():
    st.set_page_config(page_title="Image Classifier", page_icon=":camera:")
    st.title("Image Classifier")
    st.write("This app uses a pre-trained ResNet50 model to classify images into one of 1000 categories.")
    # Create a file uploader and classify the uploaded image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_class = classify_image(uploaded_file)
        # Display the top 3 predicted labels and their probabilities
        st.write(f"Top 3 predicted labels and their probabilities:")
        for label, prob in image_class:
            st.write(f"- {label}: {prob:.2f}%")

if __name__ == '__main__':
    app()
