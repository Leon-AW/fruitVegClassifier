import cv2
import tensorflow as tf
import numpy as np
import os
import time

# --- Constants ---
IMG_WIDTH = 100
IMG_HEIGHT = 100
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- Load Class Names (same logic as training script) ---
# This assumes the script is run from within fruit_classifier_project
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_base_parent_dir = os.path.join(base_dir, 'fruits-360_100x100', 'fruits-360') # Path to where Training/Test folders are
train_dir_for_classes = os.path.join(dataset_base_parent_dir, 'Training')

try:
    class_names = sorted(os.listdir(train_dir_for_classes))
    class_names = [name for name in class_names if os.path.isdir(os.path.join(train_dir_for_classes, name))]
    if not class_names:
        raise ValueError("No class subdirectories found. Cannot determine class names.")
    NUM_CLASSES = len(class_names)
    print(f"Loaded {NUM_CLASSES} class names. First few: {class_names[:5]}")
except Exception as e:
    print(f"Error loading class names: {e}")
    print("Please ensure the 'fruits-360_100x100/fruits-360/Training' directory exists relative to this script.")
    exit()

# --- Load the Trained Model ---
model_path = os.path.join(base_dir, "fruit_classifier_best.keras")
print(f"Loading model from: {model_path}")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    model.summary() # Print model summary to verify
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0) # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nStarting webcam feed. Press 'q' to quit.")

# For FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocessing for the model
    # 1. Resize
    img_resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    
    # 2. Convert BGR to RGB (OpenCV loads as BGR, model trained on RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Convert to NumPy array and expand dimensions for batch
    # Model expects float32 input in [0,255] range, rescaling is part of the model
    img_array = np.array(img_rgb, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # 4. Make prediction
    predictions = model.predict(img_batch, verbose=0) # verbose=0 to suppress progress bar
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100 # As percentage

    # FPS Calculation
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) if (new_frame_time-prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {fps:.2f}"

    # Display the prediction and confidence on the original frame (not the resized one)
    cv2.putText(frame, f"Prediction: {predicted_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Real-time Fruit Classifier', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Webcam feed stopped.") 