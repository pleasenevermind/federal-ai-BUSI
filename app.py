import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = 'Dataset_BUSI_with_GT'  # Replace with the path to your dataset

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    
    for label in ['benign', 'malignant', 'normal']:
        class_folder = os.path.join(dataset_path, label)
        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(image_path)
                # Convert to RGB if not already in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(np.array(image))
                labels.append(label)
    
    # Convert lists to NumPy arrays
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize the images
    labels = pd.get_dummies(labels).values
    
    return images, labels

# Function to create the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        # No need to specify input shape here, it will be inferred
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: benign, malignant, normal
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and save the model
def train_and_save_model(images, labels):
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    model = create_model()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    model.save('breast_cancer_model.h5')
    return model

# Function to predict the class of an image
def predict(model, image):
    prediction = model.predict(image)
    return prediction[0]

# Function to load and preprocess image
def load_and_preprocess_image(image_file):
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

# Streamlit page for image detection
def image_detection_page(model):
    st.title("Breast Ultrasound Image Analysis")
    # Description beneath the title
    st.write("""
    Welcome to our Breast Ultrasound Image Analysis Tool, a cutting-edge application 
    that leverages the power of artificial intelligence and federated learning to 
    assist in the early detection of breast cancer. By uploading a breast ultrasound 
    image, you can quickly receive a preliminary analysis that estimates the 
    likelihood of benign or malignant findings, complete with a confidence percentage.

    Our innovative approach uses federated learning, ensuring that your data privacy 
    is respected. This means that while our AI model learns from diverse datasets to 
    improve its accuracy, your personal data never leaves your device. We are 
    committed to providing a tool that not only offers insights but also upholds the 
    highest standards of privacy.
    """)

    # Warning message in yellow color
    st.warning("""
    Please note that this application is designed to be an informative resource and 
    should not replace professional medical evaluation. For a definitive diagnosis 
    and personalized medical advice, we strongly recommend consulting with your 
    healthcare provider.
    """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_and_preprocess_image(uploaded_file)
        prediction = predict(model, image)
        class_names = ['benign', 'malignant', 'normal']
        predicted_class = class_names[np.argmax(prediction)]
        
        # Define risk thresholds based on hypothetical industry standards
        high_risk_threshold = 0.75  # 75% or higher for high risk
        medium_risk_threshold = 0.50  # Between 50% and 75% for medium risk

        # Get the probability of the malignant class
        malignant_prob = prediction[1]

        # Determine the risk level and compose a user-friendly message
        if predicted_class == 'malignant':
            if malignant_prob >= high_risk_threshold:
                risk_message = f"There is a high likelihood of malignancy at {malignant_prob * 100:.2f}%. We recommend consulting with a healthcare professional for further evaluation."
            elif malignant_prob >= medium_risk_threshold:
                risk_message = f"There is a moderate likelihood of malignancy at {malignant_prob * 100:.2f}%. It's important to discuss these results with a healthcare professional."
            else:
                risk_message = f"There is a lower likelihood of malignancy at {malignant_prob * 100:.2f}%, but we still advise following up with a healthcare professional."
        elif predicted_class == 'benign':
            benign_prob = prediction[0]
            if benign_prob >= high_risk_threshold:
                risk_message = f"The likelihood of a benign condition is high at {benign_prob * 100:.2f}%. However, it's always good to have any findings evaluated by a healthcare professional."
            elif benign_prob >= medium_risk_threshold:
                risk_message = f"There is a moderate likelihood of a benign condition at {benign_prob * 100:.2f}%. Monitoring or further medical advice may be beneficial."
            else:
                risk_message = f"There is a lower likelihood of a benign condition at {benign_prob * 100:.2f}%. If you have any concerns, please consult with a healthcare provider."
        else:
            normal_prob = prediction[2]
            risk_message = f"The image analysis suggests no signs of breast cancer with a confidence of {normal_prob * 100:.2f}%. Remember to continue regular screenings as recommended by health guidelines."

        st.write(risk_message)

# # Streamlit page for training the model
# def training_page():
#     st.title("Train Breast Cancer Detection Model")
#     if st.button('Train Model'):
#         images, labels = load_dataset(DATASET_PATH)
#         model = train_and_save_model(images, labels)
#         st.write("Model trained and saved successfully!")

# Function to display data exploration page
def data_exploration_page(dataset_path):
    st.title("Data Exploration")
    
    # Count the number of images in each class
    class_counts = {}
    for class_name in ['benign', 'malignant', 'normal']:
        class_folder = os.path.join(dataset_path, class_name)
        class_counts[class_name] = len([name for name in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, name))])
    
    # Display the counts as a bar chart
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), ax=ax)
    st.pyplot(fig)
    
    # Display some sample images from each class
    st.subheader("Sample Images from Each Class")
    for class_name, count in class_counts.items():
        st.markdown(f"### {class_name.capitalize()} Samples")
        class_folder = os.path.join(dataset_path, class_name)
        sample_images = os.listdir(class_folder)[:3]  # Display 3 sample images
        cols = st.columns(3)
        for idx, image_name in enumerate(sample_images):
            image_path = os.path.join(class_folder, image_name)
            image = Image.open(image_path)
            with cols[idx]:
                st.image(image, caption=image_name, use_column_width=True)


# Main function to run the Streamlit app
def main():
    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Breast Cancer Detection using Convolutional Neural Network on Federated Learning")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Image Detection","Data Exploration"])

    # Check if the model already exists
    model_path = 'breast_cancer_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        # If the model does not exist, train and save the model
        with st.spinner('Model not found. Training a new model...'):
            images, labels = load_dataset(DATASET_PATH)
            model = train_and_save_model(images, labels)
        st.success('Model trained and saved successfully!')

    # Depending on the app mode, perform the corresponding action
    if app_mode == "Image Detection":
        image_detection_page(model)
    elif app_mode == "Data Exploration":
        data_exploration_page(DATASET_PATH)
    # elif app_mode == "Train Model":
    #     training_page()

if __name__ == "__main__":
    main()