import cv2
import torch
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array,load_img # type: ignore
from ultralytics import YOLO
import shutil

# Load YOLOv11 detection model
yolo_model = YOLO("/home/abdo/Code/Python/Deep_Learning/plant_village/training/best (2).pt")
# Load TensorFlow classification model
classifier_model = load_model("/home/abdo/Code/Python/Deep_Learning/plant_village/training/models/vgg_model_2.h5")

class_labels = ['Cucumber__Anthracnose',
 'Cucumber__Bacterial Wilt',
 'Cucumber__Downy Mildew',
 'Cucumber__Fresh Leaf',
 'Cucumber__Gummy_Stem_Blight',
 'Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Yellow_Leaf_Curl_Virus',
 'Tomato___mosaic_virus',
 'Tomato___healthy']

cropped_images_dir="Deep_Learning/plant_village/cropped_images"
if os.path.exists(cropped_images_dir):
    shutil.rmtree(cropped_images_dir)
# detect leafs and cropped them using yolo model
results=yolo_model.predict(source="Deep_Learning/plant_village/training/sample_dataset",conf=0.7)
for result in results:
    #save crops of images
    cropped_images=result.save_crop(save_dir=cropped_images_dir)

cropped_images_leaf_dir = "Deep_Learning/plant_village/cropped_images/leaf"


classified_images_dir = "Deep_Learning/plant_village/classified_images"

# Remove existing directories if they exist

if os.path.exists(classified_images_dir):
    shutil.rmtree(classified_images_dir)

# Create classified_images directory
os.makedirs(classified_images_dir)

# Create subdirectories for each class label
for label in class_labels:
    label_dir = os.path.join(classified_images_dir, label)
    os.makedirs(label_dir)


# Function to preprocess and classify images using tensorflow model
def classify_cropped_images(directory, model, class_labels,output_dir):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load and preprocess the image
            img_path = os.path.join(directory, filename)
            image = load_img(img_path, target_size=(256, 256))  # Adjust target_size as needed
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Predict the class
            prediction = model.predict(image)
            predicted_class = class_labels[prediction.argmax()]

            # Print the results
            print(f"Image: {filename}, Predicted class: {predicted_class}")

            # Load the cropped image using OpenCV
            img_cv2 = cv2.imread(img_path)

            # Put the predicted class text on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_cv2, predicted_class, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Save the image in the corresponding class directory
            output_path = os.path.join(output_dir, predicted_class, filename)
            cv2.imwrite(output_path, img_cv2)



classify_cropped_images(cropped_images_leaf_dir, classifier_model, class_labels,classified_images_dir)
