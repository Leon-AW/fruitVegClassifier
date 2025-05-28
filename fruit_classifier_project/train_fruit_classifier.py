import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import sys # ADDED
import glob # Added for finding background files
import cv2 # Added for OpenCV image processing

# --- Base Directory ---
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Configuration based on Command Line Argument ---
# Default to original size (224x224)
IMG_WIDTH_DEFAULT = 224
IMG_HEIGHT_DEFAULT = 224
DATASET_NAME_DEFAULT = 'fruits-360_original-size'
DATASET_FOLDER_DEFAULT = 'fruits-360-original-size' # Subfolder within the archive name
BACKGROUNDS_DIR_NAME_DEFAULT = 'backgrounds'
HAS_DEDICATED_VALIDATION_DEFAULT = True

# Check for '100' argument for 100x100 version
if len(sys.argv) > 1 and sys.argv[1] == '100':
    print("Using 100x100 dataset configuration.")
    IMG_WIDTH_CFG = 100
    IMG_HEIGHT_CFG = 100
    DATASET_NAME_CFG = 'fruits-360_100x100'
    DATASET_FOLDER_CFG = 'fruits-360' # Subfolder for 100x100 dataset
    BACKGROUNDS_DIR_NAME_CFG = 'backgrounds100x100'
    HAS_DEDICATED_VALIDATION_CFG = False
else:
    print("Using default (original size) dataset configuration.")
    IMG_WIDTH_CFG = IMG_WIDTH_DEFAULT
    IMG_HEIGHT_CFG = IMG_HEIGHT_DEFAULT
    DATASET_NAME_CFG = DATASET_NAME_DEFAULT
    DATASET_FOLDER_CFG = DATASET_FOLDER_DEFAULT
    BACKGROUNDS_DIR_NAME_CFG = BACKGROUNDS_DIR_NAME_DEFAULT
    HAS_DEDICATED_VALIDATION_CFG = HAS_DEDICATED_VALIDATION_DEFAULT

# --- Constants ---
IMG_WIDTH = IMG_WIDTH_CFG
IMG_HEIGHT = IMG_HEIGHT_CFG
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32 # This can be made configurable too if needed in future
BUFFER_SIZE = tf.data.AUTOTUNE

# Paths (derived from config)
# base_dir is defined above
dataset_base_dir = os.path.join(base_dir, DATASET_NAME_CFG, DATASET_FOLDER_CFG)
train_dir = os.path.join(dataset_base_dir, 'Training')
test_dir = os.path.join(dataset_base_dir, 'Test')

if HAS_DEDICATED_VALIDATION_CFG:
    validation_dir = os.path.join(dataset_base_dir, 'Validation')
else:
    print(f"No dedicated validation set for {DATASET_NAME_CFG}. Using Test set for validation during training.")
    validation_dir = test_dir # Use Test set as validation for 100x100 dataset

user_corrected_data_path = os.path.join(base_dir, "user_corrected_data")

# --- Background Images Path ---
backgrounds_dir = os.path.join(base_dir, BACKGROUNDS_DIR_NAME_CFG)
background_image_paths = glob.glob(os.path.join(backgrounds_dir, '*.jpg')) # Assumes JPEGs, add more patterns if needed e.g., '*.png'
background_image_paths.extend(glob.glob(os.path.join(backgrounds_dir, '*.png')))

if not background_image_paths:
    print(f"WARNING: No background images found in {backgrounds_dir}. Background replacement will not be effective.")
    # Optionally, you could make this an error:
    # print(f"ERROR: No background images found in {backgrounds_dir}. Please add some background images.")
    # exit()
else:
    print(f"Found {len(background_image_paths)} background images.")


# --- Determine Number of Classes ---
# The number of classes will be the number of subdirectories in the train_dir
try:
    class_names = sorted(os.listdir(train_dir))
    # Filter out any files like .DS_Store if they exist at the class level
    class_names = [name for name in class_names if os.path.isdir(os.path.join(train_dir, name))]
    NUM_CLASSES = len(class_names)
    if NUM_CLASSES == 0:
        raise ValueError("No class subdirectories found in the training directory. Check the path and dataset structure.")
    print(f"Found {NUM_CLASSES} classes using train_dir: {class_names[:5]}...") # Print first 5 for brevity
except FileNotFoundError:
    print(f"ERROR: Training directory not found at {train_dir}")
    print("Please ensure the dataset is correctly unzipped and the paths are correct.")
    exit()


# --- Background Replacement Function (Python logic) ---
def replace_background_py(image_tensor):
    """
    Replaces white background of a fruit image with a random background.
    Assumes fruit image has white background (approx R,G,B > 240).
    Args:
        image_tensor: TensorFlow tensor of the fruit image, dtype float32, range [0, 255].
    Returns:
        NumPy array of the fruit with new background, dtype float32, range [0, 255].
    """
    # Convert TensorFlow tensor to NumPy array
    image_np_0_255 = image_tensor.numpy()
    
    if not background_image_paths: # Fallback if no backgrounds
        return image_np_0_255

    # Choose a random background image
    bg_path = np.random.choice(background_image_paths)
    try:
        bg_image = cv2.imread(bg_path)
        if bg_image is None:
            print(f"Warning: Failed to load background image {bg_path}")
            return image_np_0_255 # Return original image on error
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        bg_image = cv2.resize(bg_image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Warning: Error processing background {bg_path}: {e}")
        return image_np_0_255 # Return original image on error
    
    bg_image = bg_image.astype(np.float32) # Ensure float32 for combination

    # Create mask for the fruit (non-white pixels)
    # image_np_0_255 is float32 [0, 255]. For thresholding, uint8 comparison is safer.
    img_uint8 = image_np_0_255.astype(np.uint8)
    
    # Define white:
    # A common way for Fruits-360 is to check if R, G, B are all high.
    # Or sum of pixel values.
    # Let's try a threshold on the sum. A perfect white is 255*3 = 765.
    # We can also check individual channels.
    lower_white = np.array([230, 230, 230], dtype=np.uint8) # Lower bound for white
    upper_white = np.array([255, 255, 255], dtype=np.uint8) # Upper bound for white
    
    # Create a mask where white pixels are 0 and fruit pixels are 1
    # cv2.inRange creates a binary mask: 255 for pixels in range, 0 for out of range.
    background_mask_cv = cv2.inRange(img_uint8, lower_white, upper_white)
    
    # Invert mask: fruit is 1 (or 255), background is 0
    fruit_mask_cv = cv2.bitwise_not(background_mask_cv)
    
    # Normalize mask to [0, 1] and ensure it's float32 and 3-channel for broadcasting
    fruit_mask = (fruit_mask_cv / 255.0).astype(np.float32)
    fruit_mask = np.expand_dims(fruit_mask, axis=-1) # Shape (IMG_WIDTH, IMG_HEIGHT, 1)

    # Combine: fruit * mask + background * (1 - mask)
    # image_np_0_255 is already float32
    new_image = image_np_0_255 * fruit_mask + bg_image * (1.0 - fruit_mask)
    
    # Ensure output is float32, range [0, 255]
    return new_image.astype(np.float32)

# --- TensorFlow Wrapper for the Python Function ---
def tf_replace_background(image_tensor, label):
    """
    TensorFlow wrapper for the replace_background_py function.
    image_tensor is expected to be float32, range [0, 255].
    """
    # py_function expects a list of tensors as input
    [image_with_new_bg,] = tf.py_function(
        replace_background_py, 
        [image_tensor], 
        [tf.float32] # Output type
    )
    # py_function loses shape information, so we need to set it back
    image_with_new_bg.set_shape([IMG_WIDTH, IMG_HEIGHT, 3])
    return image_with_new_bg, label


# --- Load Data ---
print("Loading main training data...")
# Ensure class_names is populated before it might be used by user_added_dataset
if not class_names: # Should have been populated by the NUM_CLASSES section
    print("Error: class_names not defined before loading data. Exiting.")
    exit()

main_train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("Loading main validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Initialize train_dataset with the main training data
train_dataset = main_train_dataset

# --- Load User Corrected Data (if available) ---
user_corrected_images_loaded = False
if os.path.exists(user_corrected_data_path) and any(os.scandir(user_corrected_data_path)):
    print(f"Found user_corrected_data directory at: {user_corrected_data_path}")
    try:
        user_class_dirs = [d.name for d in os.scandir(user_corrected_data_path) if d.is_dir()]
        if user_class_dirs:
            print(f"User corrected class directories found: {user_class_dirs[:5]}...")
            user_added_dataset = tf.keras.utils.image_dataset_from_directory(
                user_corrected_data_path,
                labels='inferred',
                label_mode='categorical',
                class_names=class_names,  # Use pre-determined class_names for consistency
                image_size=IMAGE_SIZE,
                interpolation='nearest',
                batch_size=BATCH_SIZE, # Load batched
                shuffle=True # Shuffle this dataset too
            )
            print("User corrected data loaded. Concatenating with main training data.")
            train_dataset = train_dataset.concatenate(user_added_dataset)
            user_corrected_images_loaded = True
        else:
            print("User corrected data directory is empty or has no class subdirectories. Skipping.")
    except Exception as e:
        print(f"Could not load user corrected data: {e}. Using only main training data.")
else:
    print(f"User corrected data directory not found or is empty at {user_corrected_data_path}. Using only main training data.")


# --- Apply Background Replacement (if backgrounds are available) ---
# This needs to operate on unbatched data and then re-batch.
if background_image_paths:
    print("\nApplying background replacement to training dataset...")
    train_dataset = train_dataset.unbatch() # Unbatch the (potentially concatenated) dataset
    train_dataset = train_dataset.map(tf_replace_background, num_parallel_calls=tf.data.AUTOTUNE)
    # Batching will happen after potential shuffling later
    print("Background replacement mapping applied to training dataset.")

    print("\nApplying background replacement to validation dataset...")
    validation_dataset = validation_dataset.unbatch()
    validation_dataset = validation_dataset.map(tf_replace_background, num_parallel_calls=tf.data.AUTOTUNE)
    # Batching will happen later for validation dataset as well
    print("Background replacement mapping applied to validation dataset.")
else:
    print("\nSkipping background replacement as no background images were found.")


# --- Configure dataset for performance ---
SHUFFLE_BUFFER_SIZE = 1000 # Buffer size for shuffling

# Shuffle the unbatched training data (important after concatenation and mapping)
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)

# Now batch the datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)

# Test dataset (remains separate and doesn't get these augmentations/concatenations)
print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False
).prefetch(buffer_size=BUFFER_SIZE)

print("\nData loading and preprocessing complete.")
print(f"Training dataset: {train_dataset}")
print(f"Validation dataset: {validation_dataset}")

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    layers.RandomRotation(0.1), # Rotate by a factor of 0.1 (e.g. 10% of 2*pi)
    layers.RandomZoom(0.1), # Zoom by a factor of 0.1
    # Consider adding more augmentation like RandomContrast, RandomBrightness if needed
    # layers.RandomContrast(factor=0.2),
    # layers.RandomBrightness(factor=0.2),
])

# --- Preprocessing Layer (Normalization) ---
# MobileNetV2 expects inputs in the range [-1, 1]
# We can use the built-in preprocess_input function or a Rescaling layer
# If using tf.keras.applications.mobilenet_v2.preprocess_input,
# it needs to be applied after batching and before the model.
# Alternatively, a Rescaling layer is simpler if inputs are 0-255 initially.
# image_dataset_from_directory loads images with pixel values in [0, 255].
# So, we rescale to [-1, 1].
preprocess_input_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# --- Visualization of a Sample Batch (Sanity Check) ---
# Ensure class_names is available for visualization titles
if not class_names:
    print("Warning: class_names not defined for visualization. Titles may be incorrect.")

def show_sample_batch(dataset_to_visualize, augmentation_layer, num_total_samples=16): # num_total_samples is now a cap
    """Shows a few samples from a dataset after applying augmentation."""
    try:
        # dataset_to_visualize is expected to be a dataset that yields at least one batch.
        # Example: train_dataset.take(1)
        sample_images, sample_labels = next(iter(dataset_to_visualize))
    except tf.errors.OutOfRangeError:
        print("Warning: Could not get enough samples for visualization (dataset was empty). Skipping visualization.")
        return
    except Exception as e:
        print(f"Warning: Error getting samples for visualization: {e}. Skipping visualization.")
        return
    
    augmented_images = augmentation_layer(sample_images, training=True) 

    # Determine how many images to actually show based on batch size and requested number
    num_to_show = min(num_total_samples, augmented_images.shape[0], BATCH_SIZE) # Cap at BATCH_SIZE or less if batch is smaller
    
    if num_to_show == 0:
        print("Warning: No images to show in the visualization batch.")
        return

    # Adjust subplot grid: aim for a somewhat square layout e.g. 4x4 for 16, 3x3 for 9, 2x2 for 4
    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))

    plt.figure(figsize=(cols * 3, rows * 3)) # Adjust figure size based on grid
    for i in range(num_to_show):
        ax = plt.subplot(rows, cols, i + 1)
        # Images from train_dataset are float32 [0, 255] after background replacement.
        # Augmentation layer doesn't change this range.
        plt.imshow(augmented_images[i].numpy().astype("uint8")) 
        class_index = np.argmax(sample_labels[i])
        plt.title(class_names[class_index] if class_names and class_index < len(class_names) else f"Class {class_index}", fontsize=8)
        plt.axis("off")
    
    visualization_path = os.path.join(base_dir, 'sample_augmented_batch.png')
    plt.tight_layout()
    plt.savefig(visualization_path)
    print(f"\nSaved sample augmented batch visualization to {visualization_path}")

# Show samples before training begins
# The train_dataset is already batched and should have background replacement applied.
# The data_augmentation layer will be applied by the show_sample_batch function.
print("\nVisualizing a sample batch from the processed training data (backgrounds should be replaced)...")
visualization_batch_dataset = train_dataset.take(1) # Take one batch from the fully processed train_dataset

if visualization_batch_dataset:
    # Pass the dataset (which will yield one batch) and the augmentation layer
    # num_total_samples in show_sample_batch will cap how many from the batch are shown (e.g. up to 16)
    show_sample_batch(visualization_batch_dataset, data_augmentation, num_total_samples=16)
else:
    print("Could not get a batch from train_dataset for visualization.")


# --- Build the Model ---
def build_model(num_classes, augmentation_model, preprocessing_layer):
    # Base Model: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
        include_top=False, # Exclude ImageNet classifier
        weights='imagenet'  # Load pre-trained weights
    )
    base_model.trainable = False # Freeze the base model

    # Create new model on top
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = augmentation_model(inputs)       # Apply data augmentation
    x = preprocessing_layer(x)           # Apply preprocessing (rescaling for MobileNetV2)
    x = base_model(x, training=False)  # Set training=False as base_model is frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) # Regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

print("\nBuilding model...")
model = build_model(NUM_CLASSES, data_augmentation, preprocess_input_layer)

# --- Compile the Model ---
# Using a lower learning rate for transfer learning is often beneficial
initial_learning_rate = 0.001 # Can be tuned
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("\nModel compilation complete.")

# --- Model Naming and Versioning ---
def get_next_model_version(base_filename, model_dir):
    """Finds the next available version number for a model filename."""
    version = 1
    while True:
        # Check for both .keras and older .h5 formats if necessary, though we use .keras
        potential_filename_keras = os.path.join(model_dir, f"{base_filename}_v{version}.keras")
        if not os.path.exists(potential_filename_keras):
            return version
        version += 1

version_number = get_next_model_version("fruit_classifier_best", base_dir)
checkpoint_base_name = f"fruit_classifier_best_v{version_number}"
final_model_base_name = f"fruit_classifier_final_v{version_number}"

checkpoint_path = os.path.join(base_dir, f"{checkpoint_base_name}.keras")
final_model_path = os.path.join(base_dir, f"{final_model_base_name}.keras")

print(f"\nModels will be saved with version: {version_number}")
print(f"Best model checkpoint path: {checkpoint_path}")
print(f"Final model path: {final_model_path}")


# --- Callbacks ---
# EarlyStopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3, # Number of epochs with no improvement after which training will be stopped
    verbose=1,
    restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
)

# ModelCheckpoint to save the best model
# The path should be inside 'fruit_classifier_project'
# checkpoint_path = os.path.join(base_dir, "fruit_classifier_best.keras") # Save in .keras format
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True, # Only save a model if `val_loss` has improved
    monitor='val_loss',
    verbose=1
)

# --- Train the Model ---
EPOCHS = 50 # Start with a reasonable number, EarlyStopping will handle if it's too much

print(f"\nStarting training for {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining complete.")

# --- Evaluate the Model ---
print("\nEvaluating model on the test set...")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# --- Plotting Training History (Optional but recommended) ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']  # Fixed: was 'validation_accuracy'
loss = history.history['loss']
val_loss = history.history['val_loss']     # Fixed: was 'validation_loss'
epochs_range = range(len(acc)) # Use actual number of epochs run if early stopping occurred

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
history_plot_path_acc = os.path.join(base_dir, 'training_validation_accuracy.png')
plt.savefig(history_plot_path_acc)
print(f"Saved accuracy plot to {history_plot_path_acc}")


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
history_plot_path_loss = os.path.join(base_dir, 'training_validation_loss.png')
plt.savefig(history_plot_path_loss)
print(f"Saved loss plot to {history_plot_path_loss}")
# plt.show() # Comment out if running in a non-GUI environment

# --- Save the trained model ---
# The best model is already saved by ModelCheckpoint if restore_best_weights=True was used with EarlyStopping
# and save_best_only=True with ModelCheckpoint.
# If you want to save the final state regardless:
# final_model_path = os.path.join(base_dir, "fruit_classifier_final.keras")
model.save(final_model_path)
print(f"Final model saved to {final_model_path}")
print(f"Best model (during training) saved to {checkpoint_path}")

print("\nScript finished.") 