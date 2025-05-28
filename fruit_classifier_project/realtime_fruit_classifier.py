import cv2
import tensorflow as tf
import numpy as np
import os
import time
import sys # ADDED for command-line arguments

# --- Argument Parsing ---
# Defaults
IMG_WIDTH_ARG = 224
IMG_HEIGHT_ARG = 224
MODEL_PATH_ARG = None
DATASET_FOLDER_ARG = 'fruits-360-original-size'
DATASET_ARCHIVE_NAME_ARG = 'fruits-360_original-size'

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "res" and i + 1 < len(sys.argv):
        if sys.argv[i+1] == "100":
            IMG_WIDTH_ARG = 100
            IMG_HEIGHT_ARG = 100
            DATASET_FOLDER_ARG = 'fruits-360'
            DATASET_ARCHIVE_NAME_ARG = 'fruits-360_100x100'
            print("Configuration: Using 100x100 resolution.")
        else:
            print(f"Warning: Unrecognized value '{sys.argv[i+1]}' for 'res' argument. Using default 224x224.")
        i += 2
    elif arg == "model" and i + 1 < len(sys.argv):
        MODEL_PATH_ARG = sys.argv[i+1]
        print(f"Configuration: Attempting to load specified model: {MODEL_PATH_ARG}")
        i += 2
    else:
        print(f"Warning: Unrecognized argument or missing value: {arg}")
        i += 1

# --- Constants based on Arguments ---
IMG_WIDTH = IMG_WIDTH_ARG
IMG_HEIGHT = IMG_HEIGHT_ARG
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- Load Class Names (dynamically based on selected dataset) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_train_path = os.path.join(base_dir, DATASET_ARCHIVE_NAME_ARG, DATASET_FOLDER_ARG, 'Training')

class_names = []
try:
    if not os.path.isdir(dataset_train_path):
        raise FileNotFoundError(f"Training directory for class names not found: {dataset_train_path}")
    loaded_class_names = sorted(os.listdir(dataset_train_path))
    class_names = [name for name in loaded_class_names if os.path.isdir(os.path.join(dataset_train_path, name))]
    if not class_names:
        raise ValueError(f"No class subdirectories found in {dataset_train_path}. Cannot determine class names.")
    NUM_CLASSES = len(class_names)
    print(f"Loaded {NUM_CLASSES} class names. First few: {class_names[:5]} (from {DATASET_ARCHIVE_NAME_ARG})")
except Exception as e:
    print(f"Error loading class names: {e}")
    print(f"Please ensure the dataset specified by DATASET_ARCHIVE_NAME_ARG='{DATASET_ARCHIVE_NAME_ARG}' is available.")
    exit()

# --- Load the Trained Model ---
def find_latest_model(model_dir_search, base_model_name="fruit_classifier_best_v"):
    """Finds the latest .keras model file based on version number."""
    latest_version = -1
    latest_model_path = None
    try:
        for f_name in os.listdir(model_dir_search):
            if f_name.startswith(base_model_name) and f_name.endswith(".keras"):
                try:
                    version_str = f_name[len(base_model_name):-len(".keras")]
                    version = int(version_str)
                    if version > latest_version:
                        latest_version = version
                        latest_model_path = os.path.join(model_dir_search, f_name)
                except ValueError:
                    continue # Not a correctly formatted version number
    except FileNotFoundError:
        pass # model_dir_search might not exist
    
    # Fallback if no versioned model is found, try the non-versioned name
    if not latest_model_path:
        potential_fallback = os.path.join(model_dir_search, "fruit_classifier_best.keras")
        if os.path.exists(potential_fallback):
            print(f"No versioned model found in {model_dir_search}, using fallback: {potential_fallback}")
            return potential_fallback
            
    if latest_model_path:
        print(f"Found latest model in {model_dir_search}: {latest_model_path}")
    return latest_model_path

model_to_load_path = None
if MODEL_PATH_ARG:
    # If MODEL_PATH_ARG is not absolute, assume it's relative to base_dir
    if not os.path.isabs(MODEL_PATH_ARG):
        model_to_load_path = os.path.join(base_dir, MODEL_PATH_ARG)
    else:
        model_to_load_path = MODEL_PATH_ARG
else:
    model_to_load_path = find_latest_model(base_dir)

if not model_to_load_path or not os.path.exists(model_to_load_path):
    print(f"Error: Model file not found.")
    if MODEL_PATH_ARG:
        print(f"  Specified model path: {model_to_load_path}")
    else:
        print(f"  Searched in {base_dir} for latest or fallback model.")
    print("Please ensure the model file exists or a trained model is available.")
    exit()

print(f"Loading model from: {model_to_load_path}")
try:
    model = tf.keras.models.load_model(model_to_load_path)
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