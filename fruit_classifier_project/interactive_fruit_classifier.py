import cv2
import tensorflow as tf
import numpy as np
import os
import time
import shutil # For file operations

# --- Constants ---
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
TOP_N_PREDICTIONS = 5
MAX_SEARCH_RESULTS_DISPLAY = 10 # Max number of search results to show on screen

# --- Setup Directories ---
base_dir = os.path.dirname(os.path.abspath(__file__))
user_corrected_data_dir = os.path.join(base_dir, "user_corrected_data")
if not os.path.exists(user_corrected_data_dir):
    os.makedirs(user_corrected_data_dir)
    print(f"Created directory for user corrected data: {user_corrected_data_dir}")

# --- Load Class Names ---
dataset_base_parent_dir = os.path.join(base_dir, 'fruits-360_original-size', 'fruits-360-original-size')
train_dir_for_classes = os.path.join(dataset_base_parent_dir, 'Training')
class_names = []
try:
    loaded_class_names = sorted(os.listdir(train_dir_for_classes))
    class_names = [name for name in loaded_class_names if os.path.isdir(os.path.join(train_dir_for_classes, name))]
    if not class_names:
        raise ValueError("No class subdirectories found.")
    NUM_CLASSES = len(class_names)
    print(f"Loaded {NUM_CLASSES} class names. First few: {class_names[:5]}")
except Exception as e:
    print(f"Error loading class names: {e}. Exiting.")
    exit()

# --- Load the Trained Model ---
# Attempt to load a model trained on 224x224 images.
# If you run this after training with train_fruit_classifier.py, it will look for a versioned model.
# For simplicity, let's assume you might have a generic 'fruit_classifier_best.keras' 
# or a versioned one. The training script now saves versioned models.
# We need a robust way to find the latest model.

def find_latest_model(model_dir, base_model_name="fruit_classifier_best_v"):
    """Finds the latest .keras model file based on version number."""
    latest_version = -1
    latest_model_path = None
    try:
        for f_name in os.listdir(model_dir):
            if f_name.startswith(base_model_name) and f_name.endswith(".keras"):
                try:
                    # Extract version number: fruit_classifier_best_vXX.keras
                    version_str = f_name[len(base_model_name):-len(".keras")]
                    version = int(version_str)
                    if version > latest_version:
                        latest_version = version
                        latest_model_path = os.path.join(model_dir, f_name)
                except ValueError:
                    continue # Not a correctly formatted version number
    except FileNotFoundError:
        pass # model_dir might not exist yet
    
    # Fallback if no versioned model is found, try the non-versioned name
    if not latest_model_path:
        potential_fallback = os.path.join(model_dir, "fruit_classifier_best.keras")
        if os.path.exists(potential_fallback):
            print(f"No versioned model found, using fallback: {potential_fallback}")
            return potential_fallback
            
    if latest_model_path:
        print(f"Found latest model: {latest_model_path}")
    return latest_model_path

model_path = find_latest_model(base_dir)

if not model_path or not os.path.exists(model_path):
    print(f"Error: Model file not found. Searched in {base_dir}.")
    print("Please ensure you have trained a model using train_fruit_classifier.py first.")
    print("It should be named like 'fruit_classifier_best_vX.keras' or 'fruit_classifier_best.keras'.")
    exit()

print(f"Loading model from: {model_path}")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Exiting.")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit()

print("\nInteractive Fruit Classifier Started.")
print("Press 'c' to classify the current frame.")
print("Press 'q' to quit.")

current_predictions_on_frame = [] # To store top N predictions for display
last_classified_frame = None

# --- States for UI ---
STATE_IDLE = 0
STATE_SHOWING_TOP_PREDICTIONS = 1
STATE_SEARCH_INPUT = 2
STATE_SHOWING_SEARCH_RESULTS = 3
current_ui_state = STATE_IDLE

current_search_query = ""
search_results_on_frame = [] # To store search results for display

def save_image_with_label(frame_to_save, correct_class_name, user_data_dir):
    print(f"Feedback received: Correct label is '{correct_class_name}'. Saving image.")

    # Resize the frame to the target size (e.g., 100x100) before saving
    # IMAGE_SIZE is defined as (IMG_WIDTH, IMG_HEIGHT) globally
    resized_for_saving = cv2.resize(frame_to_save, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    class_specific_dir = os.path.join(user_data_dir, correct_class_name)
    if not os.path.exists(class_specific_dir):
        os.makedirs(class_specific_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Count existing files to make a unique name, more robust than just +1
    existing_files = [f for f in os.listdir(class_specific_dir) if f.startswith(correct_class_name.replace(' ', '_'))]
    image_filename = f"{correct_class_name.replace(' ', '_')}_{timestamp}_{len(existing_files)+1}.png"
    image_save_path = os.path.join(class_specific_dir, image_filename)
    
    cv2.imwrite(image_save_path, resized_for_saving) # Save the resized image
    print(f"Image saved to: {image_save_path} (resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    display_frame = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    # --- UI Display Logic ---
    y_offset = 30 # Initial y_offset for the first line of text

    if current_ui_state == STATE_IDLE:
        cv2.putText(display_frame, "Press 'c' to classify", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2) # Increased scale
    elif current_ui_state == STATE_SHOWING_TOP_PREDICTIONS:
        cv2.putText(display_frame, "Top Predictions (1-5, 'f'-search, 's'-skip):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2) # Increased scale
        y_offset += 35 # Increased offset for larger font
        for i, (class_name, confidence) in enumerate(current_predictions_on_frame):
            text = f"{i+1}: {class_name} ({confidence:.2f}%)"
            cv2.putText(display_frame, text, (10, y_offset + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Increased scale & offset
    elif current_ui_state == STATE_SEARCH_INPUT:
        cv2.putText(display_frame, f"Search (Enter, Esc): {current_search_query}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2) # Increased scale, shortened text
        y_offset += 35
        cv2.putText(display_frame, "Type query, Enter to find, Esc to cancel", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1) # Smaller helper text
    elif current_ui_state == STATE_SHOWING_SEARCH_RESULTS:
        cv2.putText(display_frame, f"Results for '{current_search_query}' (1-N, 's'-skip, Esc-back):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2) # Increased scale
        y_offset += 35 # Increased offset
        for i, class_name in enumerate(search_results_on_frame):
            if i >= MAX_SEARCH_RESULTS_DISPLAY: break
            text = f"{i+1}: {class_name}"
            cv2.putText(display_frame, text, (10, y_offset + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Increased scale & offset

    cv2.imshow('Interactive Fruit Classifier', display_frame)

    # --- Key Handling Logic ---
    if key == ord('q'):
        break
    
    if current_ui_state == STATE_IDLE:
        if key == ord('c'):
            last_classified_frame = frame.copy()
            img_resized = cv2.resize(last_classified_frame, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = np.array(img_rgb, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)
            predictions_raw = model.predict(img_batch, verbose=0)[0]
            
            top_indices = np.argsort(predictions_raw)[-TOP_N_PREDICTIONS:][::-1]
            current_predictions_on_frame = []
            for i in top_indices:
                current_predictions_on_frame.append((class_names[i], predictions_raw[i] * 100))
            
            print("--- Classified --- Top Predictions:")
            for i, (name, conf) in enumerate(current_predictions_on_frame):
                print(f"{i+1}: {name} ({conf:.2f}%)")
            print("Enter 1-5 to select, 'f' to search, or 's' to skip.")
            current_ui_state = STATE_SHOWING_TOP_PREDICTIONS

    elif current_ui_state == STATE_SHOWING_TOP_PREDICTIONS:
        if ord('1') <= key <= ord(str(min(TOP_N_PREDICTIONS, len(current_predictions_on_frame)))):
            selected_index = int(chr(key)) - 1
            correct_class_name, _ = current_predictions_on_frame[selected_index]
            save_image_with_label(last_classified_frame, correct_class_name, user_corrected_data_dir)
            current_ui_state = STATE_IDLE
            current_predictions_on_frame = []
            last_classified_frame = None
        elif key == ord('f'):
            current_ui_state = STATE_SEARCH_INPUT
            current_search_query = ""
            search_results_on_frame = []
            print("Switched to search mode. Type your query and press Enter.")
        elif key == ord('s'):
            print("Skipped labeling.")
            current_ui_state = STATE_IDLE
            current_predictions_on_frame = []
            last_classified_frame = None

    elif current_ui_state == STATE_SEARCH_INPUT:
        if key == 27: # Escape key
            current_ui_state = STATE_SHOWING_TOP_PREDICTIONS # Go back to showing top predictions
            current_search_query = ""
        elif key == 13 or key == 10: # Enter key (13 for Windows, 10 for Unix-like)
            if current_search_query:
                print(f"Searching for: '{current_search_query}'")
                search_results_on_frame = [name for name in class_names if current_search_query.lower() in name.lower()]
                if search_results_on_frame:
                    print("--- Search Results ---")
                    for i, name in enumerate(search_results_on_frame):
                        if i >= MAX_SEARCH_RESULTS_DISPLAY: 
                            print(f"...and {len(search_results_on_frame) - MAX_SEARCH_RESULTS_DISPLAY} more.")
                            break
                        print(f"{i+1}: {name}")
                    current_ui_state = STATE_SHOWING_SEARCH_RESULTS
                else:
                    print("No results found.") # Stay in search input or go back to top 5?
                    # For now, let's stay in search input, user can Esc or refine search
            else:
                print("Empty search query. Type something or Esc to cancel.")
        elif 32 <= key <= 126: # Printable ASCII characters
            current_search_query += chr(key)
        elif key == 8: # Backspace
            current_search_query = current_search_query[:-1]
    
    elif current_ui_state == STATE_SHOWING_SEARCH_RESULTS:
        # Determine selected_index from key press
        char_key = chr(key)
        selected_index = -1 # Default to no valid selection
        
        # Determine how many results are actually shown on screen (up to MAX_SEARCH_RESULTS_DISPLAY)
        num_results_shown_on_screen = min(len(search_results_on_frame), MAX_SEARCH_RESULTS_DISPLAY)

        if '1' <= char_key <= '9':
            digit_pressed = int(char_key)
            # Check if the digit pressed is within the range of items displayed
            if digit_pressed <= num_results_shown_on_screen:
                selected_index = digit_pressed - 1 # 0-indexed
        elif char_key == '0':
            # '0' key is used for the 10th item, if 10 items are shown
            if num_results_shown_on_screen == 10:
                selected_index = 9 # 0-indexed for the 10th item
        
        # Action branches based on the key press
        if selected_index != -1: # A valid numeric item (1-9, or 0 for 10th) was selected
            correct_class_name = search_results_on_frame[selected_index]
            save_image_with_label(last_classified_frame, correct_class_name, user_corrected_data_dir)
            
            # Reset state to idle
            current_ui_state = STATE_IDLE
            search_results_on_frame = []
            current_search_query = ""
            last_classified_frame = None
        elif key == ord('s'): # Skip labeling
            print("Skipped labeling from search results.")
            current_ui_state = STATE_IDLE
            search_results_on_frame = []
            current_search_query = ""
            last_classified_frame = None
        elif key == 27: # Escape key - go back to showing top predictions
            current_ui_state = STATE_SHOWING_TOP_PREDICTIONS 
            # Clear only search-specific state, keep top predictions and classified frame
            search_results_on_frame = []
            current_search_query = ""
            print("Cancelled search. Returning to top predictions.") # Added print for clarity

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Classifier stopped.") 