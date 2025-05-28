import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import sys # ADDED
import glob # Added for finding background files
import cv2 # Added for OpenCV image processing

# --- GPU/MPS Configuration ---
print("TensorFlow version:", tf.__version__)

# Configure GPU/MPS usage
gpus = tf.config.list_physical_devices('GPU')
mps_devices = tf.config.list_physical_devices('MPS') if hasattr(tf.config, 'list_physical_devices') else []

device_info = "CPU"
if gpus:
    try:
        # Configure GPU memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        device_info = f"GPU ({len(gpus)} device(s))"
        print(f"GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        device_info = "CPU (GPU config failed)"
elif mps_devices:
    try:
        # Apple Silicon MPS (Metal Performance Shaders) support
        device_info = "MPS (Apple Silicon GPU)"
        print("MPS (Apple Silicon GPU) device found and will be used for training.")
        print("Note: MPS provides GPU acceleration on Apple Silicon Macs (M1/M2/M3/M4).")
    except RuntimeError as e:
        print(f"MPS configuration error: {e}")
        device_info = "CPU (MPS config failed)"
else:
    print("No GPU or MPS devices found. Using CPU for training.")
    device_info = "CPU"

print(f"Training will use: {device_info}")

# Set mixed precision policy for better performance on compatible hardware
if gpus or mps_devices:
    try:
        # Mixed precision can significantly speed up training on modern GPUs and Apple Silicon
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision (float16) enabled for faster training.")
    except Exception as e:
        print(f"Mixed precision setup failed: {e}. Using default float32.")

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
MODEL_ARCH_DEFAULT = 'mobilenet'

# Parse command line arguments
IMG_WIDTH_CFG = IMG_WIDTH_DEFAULT
IMG_HEIGHT_CFG = IMG_HEIGHT_DEFAULT
DATASET_NAME_CFG = DATASET_NAME_DEFAULT
DATASET_FOLDER_CFG = DATASET_FOLDER_DEFAULT
BACKGROUNDS_DIR_NAME_CFG = BACKGROUNDS_DIR_NAME_DEFAULT
HAS_DEDICATED_VALIDATION_CFG = HAS_DEDICATED_VALIDATION_DEFAULT
MODEL_ARCH_CFG = MODEL_ARCH_DEFAULT

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '100':
        print("Using 100x100 dataset configuration.")
        IMG_WIDTH_CFG = 100
        IMG_HEIGHT_CFG = 100
        DATASET_NAME_CFG = 'fruits-360_100x100'
        DATASET_FOLDER_CFG = 'fruits-360' 
        BACKGROUNDS_DIR_NAME_CFG = 'backgrounds100x100'
        HAS_DEDICATED_VALIDATION_CFG = False
        i += 1
    elif arg == 'arch' and i + 1 < len(sys.argv):
        if sys.argv[i+1].lower() == 'efficientnet':
            MODEL_ARCH_CFG = 'efficientnet'
            print("Using EfficientNetB3 model architecture.")
        elif sys.argv[i+1].lower() == 'mobilenet':
            MODEL_ARCH_CFG = 'mobilenet'
            print("Using MobileNetV2 model architecture.")
        else:
            print(f"Warning: Unrecognized architecture '{sys.argv[i+1]}'. Using default MobileNetV2.")
        i += 2
    else:
        print(f"Warning: Unrecognized argument: {arg}")
        i += 1

if len(sys.argv) == 1:
    print("Using default (original size) dataset configuration.")
    print("Using default MobileNetV2 model architecture.")

IMG_WIDTH = IMG_WIDTH_CFG
IMG_HEIGHT = IMG_HEIGHT_CFG
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32 # This can be made configurable too if needed in future
BUFFER_SIZE = tf.data.AUTOTUNE
MODEL_ARCH = MODEL_ARCH_CFG

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
if not class_names: 
    print("Error: class_names not defined before loading data. Exiting.")
    exit()

main_train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE, # Loaded batched initially
    shuffle=True,
    seed=42
)

# --- Prepare main_train_dataset: unbatch and cast to float32 BEFORE concatenation ---
print("Preparing main training data (unbatching and casting to float32)...")
main_train_dataset_unbatched = main_train_dataset.unbatch()
def cast_image_to_float32(image, label):
    return tf.cast(image, tf.float32), label
main_train_dataset_unbatched_float32 = main_train_dataset_unbatched.map(cast_image_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
# main_train_dataset is now unbatched and has float32 images.

print("Loading main validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE, # Loaded batched initially
    shuffle=False
)

# --- Prepare validation_dataset: unbatch and cast to float32 ---
print("Preparing validation data (unbatching and casting to float32)...")
validation_dataset_unbatched = validation_dataset.unbatch()
validation_dataset_unbatched_float32 = validation_dataset_unbatched.map(cast_image_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
# validation_dataset is now unbatched and has float32 images.

# Initialize train_dataset with the processed main training data
train_dataset = main_train_dataset_unbatched_float32 

# Initialize validation_dataset with the processed main validation data
# This will be the base for validation, to which user-corrected validation samples are added
current_validation_dataset = validation_dataset_unbatched_float32

# --- Load User Corrected Data (if available) and split for train/validation ---
user_corrected_images_loaded_to_train = False
user_corrected_images_loaded_to_val = False

if os.path.exists(user_corrected_data_path) and any(os.scandir(user_corrected_data_path)):
    print(f"Found user_corrected_data directory at: {user_corrected_data_path}")
    try:
        user_class_dirs = [d.name for d in os.scandir(user_corrected_data_path) if d.is_dir()]
        if user_class_dirs:
            print(f"User corrected class directories found: {user_class_dirs[:5]}...")
            
            user_train_image_paths = []
            user_train_labels = []
            user_val_image_paths = []
            user_val_labels = []
            
            for class_dir_name in user_class_dirs:
                class_dir_path = os.path.join(user_corrected_data_path, class_dir_name)
                if class_dir_name not in class_names:
                    print(f"Warning: User corrected class '{class_dir_name}' not found in main dataset classes. Skipping.")
                    continue
                
                class_index = class_names.index(class_dir_name)
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
                class_image_paths_for_class = []
                for ext in image_extensions:
                    class_image_paths_for_class.extend(glob.glob(os.path.join(class_dir_path, ext)))
                    class_image_paths_for_class.extend(glob.glob(os.path.join(class_dir_path, ext.upper())))
                
                if class_image_paths_for_class:
                    np.random.shuffle(class_image_paths_for_class) # Shuffle before splitting
                    num_images_in_class = len(class_image_paths_for_class)
                    
                    # Split: ~20% for validation, rest for training (per class)
                    num_val_samples_for_class = round(num_images_in_class * 0.2)
                    num_train_samples_for_class = num_images_in_class - num_val_samples_for_class

                    current_class_train_paths = class_image_paths_for_class[:num_train_samples_for_class]
                    current_class_val_paths = class_image_paths_for_class[num_train_samples_for_class:]
                    
                    if current_class_train_paths:
                        user_train_image_paths.extend(current_class_train_paths)
                        user_train_labels.extend([class_index] * len(current_class_train_paths))
                        print(f"  Added {len(current_class_train_paths)} images from '{class_dir_name}' to user training set.")
                    
                    if current_class_val_paths:
                        user_val_image_paths.extend(current_class_val_paths)
                        user_val_labels.extend([class_index] * len(current_class_val_paths))
                        print(f"  Added {len(current_class_val_paths)} images from '{class_dir_name}' to user validation set.")
            
            # Function to load and preprocess user images (common for train and val splits)
            def load_and_preprocess_user_image(image_path, label):
                image = tf.io.read_file(image_path)
                image = tf.image.decode_image(image, channels=3, expand_animations=False)
                image = tf.cast(image, tf.float32) 
                image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
                image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
                label_one_hot = tf.one_hot(label, NUM_CLASSES)
                label_one_hot.set_shape([NUM_CLASSES])
                return image, label_one_hot

            if user_train_image_paths:
                print(f"Total user corrected images for training: {len(user_train_image_paths)}")
                user_train_dataset = tf.data.Dataset.from_tensor_slices((user_train_image_paths, user_train_labels))
                user_train_dataset = user_train_dataset.map(load_and_preprocess_user_image, num_parallel_calls=tf.data.AUTOTUNE)
                user_train_dataset = user_train_dataset.shuffle(len(user_train_image_paths))
                train_dataset = train_dataset.concatenate(user_train_dataset)
                user_corrected_images_loaded_to_train = True
                print("User corrected training data concatenated.")
            else:
                print("No valid user corrected images found for the training split.")

            if user_val_image_paths:
                print(f"Total user corrected images for validation: {len(user_val_image_paths)}")
                user_val_dataset = tf.data.Dataset.from_tensor_slices((user_val_image_paths, user_val_labels))
                user_val_dataset = user_val_dataset.map(load_and_preprocess_user_image, num_parallel_calls=tf.data.AUTOTUNE)
                # Shuffling validation part of user data is optional, but good for consistency before potential batching
                user_val_dataset = user_val_dataset.shuffle(len(user_val_image_paths))
                current_validation_dataset = current_validation_dataset.concatenate(user_val_dataset) 
                user_corrected_images_loaded_to_val = True
                print("User corrected validation data concatenated.")
            else:
                print("No valid user corrected images found for the validation split.")
        else:
            print("User corrected data directory is empty or has no class subdirectories. Skipping user data.")
    except Exception as e:
        print(f"Could not load or split user corrected data: {e}. Using only main dataset.")
        import traceback
        traceback.print_exc()
else:
    print(f"User corrected data directory not found or is empty at {user_corrected_data_path}. Skipping user data.")

# Reassign the potentially augmented validation_dataset
validation_dataset = current_validation_dataset

# --- Apply Background Replacement (if backgrounds are available) ---
# train_dataset is already unbatched and has float32 images.
# validation_dataset is also unbatched and has float32 images.
if background_image_paths:
    print("\nApplying background replacement to training dataset...")
    # train_dataset = train_dataset.unbatch() # This was the redundant call, removed
    train_dataset = train_dataset.map(tf_replace_background, num_parallel_calls=tf.data.AUTOTUNE)
    print("Background replacement mapping applied to training dataset.")

    print("\nApplying background replacement to validation dataset...")
    # validation_dataset = validation_dataset.unbatch() # Already unbatched
    # validation_dataset = validation_dataset.map(cast_image_to_float32, num_parallel_calls=tf.data.AUTOTUNE) # Already float32
    validation_dataset = validation_dataset.map(tf_replace_background, num_parallel_calls=tf.data.AUTOTUNE)
    print("Background replacement mapping applied to validation dataset.")
else:
    print("\nSkipping background replacement as no background images were found.")


# --- Configure dataset for performance ---
SHUFFLE_BUFFER_SIZE = 1000 
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE).repeat()
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE).repeat()

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
# Different architectures require different preprocessing
if MODEL_ARCH == 'efficientnet':
    # EfficientNet expects inputs in the range [0, 1]
    preprocess_input_layer = tf.keras.layers.Rescaling(1./255.0)
    print("Using EfficientNet preprocessing (rescaling to [0, 1]).")
else:
    # MobileNetV2 expects inputs in the range [-1, 1]
    preprocess_input_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
    print("Using MobileNet preprocessing (rescaling to [-1, 1]).")

# --- Visualization of a Sample Batch (Sanity Check) ---
if not class_names:
    print("Warning: class_names not defined for visualization. Titles may be incorrect.")

def show_sample_batch(dataset_to_visualize, augmentation_layer, num_total_samples=16):
    """Shows a few samples from a dataset after applying augmentation."""
    try:
        sample_images, sample_labels = next(iter(dataset_to_visualize))
    except tf.errors.OutOfRangeError:
        print("Warning: Could not get enough samples for visualization (dataset was empty). Skipping visualization.")
        return
    except Exception as e:
        print(f"Warning: Error getting samples for visualization: {e}. Skipping visualization.")
        return
    
    augmented_images = augmentation_layer(sample_images, training=True) 
    num_to_show = min(num_total_samples, augmented_images.shape[0], BATCH_SIZE)
    
    if num_to_show == 0:
        print("Warning: No images to show in the visualization batch.")
        return

    cols = int(np.ceil(np.sqrt(num_to_show)))
    rows = int(np.ceil(num_to_show / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(num_to_show):
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8")) 
        class_index = np.argmax(sample_labels[i])
        plt.title(class_names[class_index] if class_names and class_index < len(class_names) else f"Class {class_index}", fontsize=8)
        plt.axis("off")
    
    visualization_path = os.path.join(base_dir, 'sample_augmented_batch.png')
    plt.tight_layout()
    plt.savefig(visualization_path)
    print(f"\nSaved sample augmented batch visualization to {visualization_path}")

print("\nVisualizing a sample batch from the processed training data (backgrounds should be replaced)...")
visualization_batch_dataset = train_dataset.take(1)

if visualization_batch_dataset:
    show_sample_batch(visualization_batch_dataset, data_augmentation, num_total_samples=16)
else:
    print("Could not get a batch from train_dataset for visualization.")

# --- Build the Model ---
def build_model(num_classes, augmentation_model, preprocessing_layer, architecture='mobilenet'):
    """
    Build a transfer learning model with the specified architecture.
    
    Args:
        num_classes: Number of output classes
        augmentation_model: Data augmentation layers
        preprocessing_layer: Preprocessing layer for the chosen architecture
        architecture: Either 'mobilenet' or 'efficientnet'
    """
    if architecture == 'efficientnet':
        # EfficientNetB3
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
            include_top=False,
            weights='imagenet'
        )
        print(f"Using EfficientNetB3 with input shape ({IMG_WIDTH}, {IMG_HEIGHT}, 3)")
    else:
        # MobileNetV2 (default)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
            include_top=False,
            weights='imagenet'
        )
        print(f"Using MobileNetV2 with input shape ({IMG_WIDTH}, {IMG_HEIGHT}, 3)")
    
    base_model.trainable = False  # Freeze the base model

    # Create new model on top
    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = augmentation_model(inputs)       # Apply data augmentation
    x = preprocessing_layer(x)           # Apply preprocessing
    x = base_model(x, training=False)    # Set training=False as base_model is frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)           # Regularization
    
    # Ensure final layer uses float32 for numerical stability with mixed precision
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

print(f"\nBuilding model with {MODEL_ARCH} architecture...")
model = build_model(NUM_CLASSES, data_augmentation, preprocess_input_layer, MODEL_ARCH)

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
        potential_filename_keras = os.path.join(model_dir, f"{base_filename}_v{version}.keras")
        if not os.path.exists(potential_filename_keras):
            return version
        version += 1

# Include architecture in model filename for clarity
model_base_name = f"fruit_classifier_{MODEL_ARCH}_best"
version_number = get_next_model_version(model_base_name, base_dir)
checkpoint_base_name = f"{model_base_name}_v{version_number}"
final_model_base_name = f"fruit_classifier_{MODEL_ARCH}_final_v{version_number}"

checkpoint_path = os.path.join(base_dir, f"{checkpoint_base_name}.keras")
final_model_path = os.path.join(base_dir, f"{final_model_base_name}.keras")

print(f"\nModels will be saved with version: {version_number}")
print(f"Best model checkpoint path: {checkpoint_path}")
print(f"Final model path: {final_model_path}")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

EPOCHS = 50
print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Device: {device_info}")
if gpus or mps_devices:
    print("Mixed precision training enabled - this should speed up training significantly.")
    print("Note: Training time will depend on dataset size, model complexity, and hardware.")

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining complete.")

print("\nEvaluating model on the test set...")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title(f'Training and Validation Accuracy ({MODEL_ARCH.upper()})')
history_plot_path_acc = os.path.join(base_dir, f'training_validation_accuracy_{MODEL_ARCH}.png')
plt.savefig(history_plot_path_acc)
print(f"Saved accuracy plot to {history_plot_path_acc}")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'Training and Validation Loss ({MODEL_ARCH.upper()})')
history_plot_path_loss = os.path.join(base_dir, f'training_validation_loss_{MODEL_ARCH}.png')
plt.savefig(history_plot_path_loss)

model.save(final_model_path)
print(f"Final model saved to {final_model_path}")
print(f"Best model (during training) saved to {checkpoint_path}")

print("\nScript finished.") 