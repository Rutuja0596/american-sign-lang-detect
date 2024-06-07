import cv2
import numpy as np
import tensorflow as tf
import json

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Rutuja/Desktop/sign-lang-detect/best_model.h5')

# Load class labels
with open('C:/Users/Rutuja/Desktop/sign-lang-detect/class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame: resize and normalize
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Make prediction
    predictions = model.predict(input_frame)
    predicted_label = class_labels[np.argmax(predictions)]

    # Display the label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
