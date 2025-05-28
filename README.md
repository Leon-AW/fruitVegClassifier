# Real-time Fruit Classification System with User Feedback Loop

## 1. Overview

This project implements a real-time fruit classification system using a deep learning model. It leverages the webcam of a MacBook (or any compatible system) to classify fruits in real-time. A key feature of this system is its interactive mode, which allows users to provide corrections for misclassifications. This feedback is then used to augment the dataset, enabling the model to be retrained and improved over time. The system supports different image resolutions and uses MobileNetV2 for efficient transfer learning.

## 2. Features

*   **Real-time Fruit Classification:** Classifies fruits from a live webcam feed.
*   **Interactive Feedback Mode:**
    *   Displays top N predictions for a captured frame.
    *   Allows users to select the correct label if the top prediction is wrong.
    *   Includes a search functionality to find the correct label from the full list of classes.
*   **User-Corrected Data Collection:** Saves user-corrected images (at the model's input resolution) into a structured directory (`user_corrected_data/<ClassName>/image.png`).
*   **Model Retraining:** The training script can incorporate images from the `user_corrected_data` directory to fine-tune and improve the model.
*   **Dataset Flexibility:**
    *   Supports the original high-resolution Fruits-360 dataset (processed at 224x224).
    *   Supports the 100x100 pixel version of the Fruits-360 dataset.
*   **Background Augmentation:** Replaces the plain white backgrounds of the Fruits-360 dataset with random background images to improve model generalization to real-world webcam images.
*   **Transfer Learning:** Uses a pre-trained MobileNetV2 model with a custom classification head.
*   **Model Versioning:** Saves trained models with version numbers (e.g., `fruit_classifier_best_v1.keras`) to keep track of improvements.
*   **Command-Line Configurability:** Scripts accept arguments for resolution and specific model loading, enhancing flexibility.

## 3. Dataset

This project uses the **Fruits-360 dataset** available on Kaggle.

*   **Kaggle Link:** [Fruits-360 Dataset](https://www.kaggle.com/datasets/moltean/fruits)

You will need to download two versions for full functionality as described:
1.  **Original Size:** The archive named something like `fruits-360_original_size.zip`.
2.  **100x100 Pixels:** The archive named `fruits-360_dataset.zip` (this one contains the 100x100 images).

### Expected Dataset Directory Structure:

After downloading and unzipping, the project expects the datasets to be organized within the `fruit_classifier_project` directory as follows:

```
fruit_detection_system/
├── fruit_classifier_project/
│   ├── fruits-360_original-size/
│   │   └── fruits-360-original-size/  # This is the actual dataset folder from the archive
│   │       ├── Training/
│   │       ├── Validation/
│   │       └── Test/
│   ├── fruits-360_100x100/
│   │   └── fruits-360/               # This is the actual dataset folder from the archive
│   │       ├── Training/
│   │       └── Test/
│   │       └── (No dedicated Validation folder, Test set is used)
│   ├── backgrounds/                  # For 224x224 background replacement
│   │   └── *.jpg, *.png
│   ├── backgrounds100x100/           # For 100x100 background replacement
│   │   └── *.jpg, *.png
│   ├── user_corrected_data/          # Will be created automatically
│   │   └── ClassName1/
│   │       └── image1.png
│   │   └── ClassName2/
│   │       └── image2.png
│   ├── train_fruit_classifier.py
│   ├── interactive_fruit_classifier.py
│   └── realtime_fruit_classifier.py
├── fruit_detector/                   # Python virtual environment (recommended)
└── README.md
```

## 4. Setup and Installation

### Prerequisites
*   Python 3.8+
*   Access to a webcam

### Steps

1.  **Clone the Repository (if applicable):**
    If this project were on a remote Git server:
    ```bash
    git clone <repository_url>
    cd fruit_detection_system
    ```
    For now, you have it locally.

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv fruit_detector
    source fruit_detector/bin/activate  # On macOS/Linux
    # For Windows: .\fruit_detector\Scripts\activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` file should be present or created with the necessary packages.
    ```bash
    pip install tensorflow matplotlib numpy opencv-python
    ```
    (If a `requirements.txt` file is provided, use `pip install -r requirements.txt`)

4.  **Download and Prepare Datasets:**
    *   Download the "original size" and "100x100" versions of the Fruits-360 dataset from Kaggle (see section 3).
    *   Unzip them into the `fruit_classifier_project/` directory as shown in the "Expected Dataset Directory Structure" above.
        *   Ensure the folder from the "original size" archive is named `fruits-360-original-size`.
        *   Ensure the folder from the "100x100" archive is named `fruits-360`.

5.  **Prepare Background Images:**
    *   Create two folders inside `fruit_classifier_project/`:
        *   `backgrounds`: For the original size (224x224) processing.
        *   `backgrounds100x100`: For the 100x100 processing.
    *   Place several `.jpg` or `.png` images into each of these folders. These will be used to replace the white backgrounds of the fruit images during training. The more varied, the better.

## 5. Directory Structure Overview

```
fruit_detection_system/
├── .git/                     # Git repository data
├── .gitignore                # Specifies intentionally untracked files
├── fruit_classifier_project/ # Main project folder
│   ├── fruits-360_original-size/ # Original size dataset
│   ├── fruits-360_100x100/     # 100x100 dataset
│   ├── backgrounds/            # Background images for 224x224 augmentation
│   ├── backgrounds100x100/     # Background images for 100x100 augmentation
│   ├── user_corrected_data/    # Stores images saved via interactive feedback
│   ├── train_fruit_classifier.py # Script for training/retraining the model
│   ├── interactive_fruit_classifier.py # Script for interactive classification and feedback
│   ├── realtime_fruit_classifier.py  # Script for simple real-time classification display
│   ├── *.keras                 # Saved models will appear here (e.g., fruit_classifier_best_v1.keras)
│   └── *.png                   # Training plots and sample augmentations will be saved here
├── fruit_detector/           # Python virtual environment
└── README.md                 # This file
```

## 6. Usage

Ensure your virtual environment is activated (`source fruit_detector/bin/activate`) before running any scripts. All commands should be run from the `fruit_detection_system` root directory.

### 6.1. Training the Model (`train_fruit_classifier.py`)

This script trains the classification model. It can use either the original-size dataset (default, 224x224) or the 100x100 dataset. It also incorporates data from `user_corrected_data/` if available.

*   **Default (Original Size - 224x224):**
    ```bash
    python fruit_classifier_project/train_fruit_classifier.py
    ```

*   **100x100 Dataset:**
    ```bash
    python fruit_classifier_project/train_fruit_classifier.py 100
    ```

**Outputs:**
*   Trained models (e.g., `fruit_classifier_best_v1.keras`, `fruit_classifier_final_v1.keras`) saved in `fruit_classifier_project/`.
*   Training history plots (accuracy and loss) saved as PNG files in `fruit_classifier_project/`.
*   A sample batch of augmented images (`sample_augmented_batch.png`) saved in `fruit_classifier_project/`.

### 6.2. Interactive Classification & Data Collection (`interactive_fruit_classifier.py`)

This script provides a webcam feed where you can classify fruits on demand and provide corrections.

*   **Default (Original Size - 224x224, latest model):**
    ```bash
    python fruit_classifier_project/interactive_fruit_classifier.py
    ```

*   **With Options:**
    *   `res 100`: Use 100x100 resolution for processing and loads class names from the 100x100 dataset structure. User-corrected images will be saved as 100x100.
    *   `model <model_name.keras>`: Load a specific Keras model file. Path can be relative to `fruit_classifier_project/` or absolute.

    **Examples:**
    ```bash
    # Use 100x100 resolution, latest 100x100 model (assumed to be compatible)
    python fruit_classifier_project/interactive_fruit_classifier.py res 100

    # Use default resolution with a specific model
    python fruit_classifier_project/interactive_fruit_classifier.py model fruit_classifier_best_v2.keras

    # Use 100x100 resolution with a specific model
    python fruit_classifier_project/interactive_fruit_classifier.py res 100 model fruit_classifier_100x100_v1.keras
    ```

**Key Controls in Interactive Mode:**
*   **`c`**: Capture the current frame and classify it. Displays top 5 predictions.
*   **`1` - `5`**: If predictions are shown, select the corresponding prediction as the correct label. The original full-resolution frame (resized to the model's input size) is saved to `user_corrected_data/ClassName/`.
*   **`s`**: Skip labeling the current classification.
*   **`f`**: After classifying (predictions are shown), press 'f' to enter search mode.
    *   **In Search Mode:** Type your search query for a fruit name.
    *   **`Enter`**: Perform the search based on your query.
    *   **`1` - `N` (up to 0 for 10th)**: Select the correct label from the filtered search results.
    *   **`Esc`**: Exit search mode (or back from search results to predictions).
*   **`q`**: Quit the application.

### 6.3. Real-time Classification Display (`realtime_fruit_classifier.py`)

This script provides a continuous real-time classification of fruits from the webcam feed, displaying the top prediction.

*   **Default (Original Size - 224x224, latest model):**
    ```bash
    python fruit_classifier_project/realtime_fruit_classifier.py
    ```

*   **With Options (same as `interactive_fruit_classifier.py`):**
    ```bash
    # Use 100x100 resolution
    python fruit_classifier_project/realtime_fruit_classifier.py res 100

    # Use a specific model
    python fruit_classifier_project/realtime_fruit_classifier.py model fruit_classifier_best_v1.keras
    ```

**Key Controls:**
*   **`q`**: Quit the application.

### 6.4. Retraining Workflow

1.  Train an initial model using `train_fruit_classifier.py`.
2.  Use `interactive_fruit_classifier.py` to classify fruits. When misclassifications occur, use the feedback mechanism (selecting the correct label or searching) to save the corrected image and label to the `user_corrected_data/` directory.
3.  Collect a sufficient amount of corrected data.
4.  Re-run `train_fruit_classifier.py`. The script will automatically detect and include images from `user_corrected_data/` in the training process, further fine-tuning the model.

## 7. Model Architecture

*   **Base Model:** MobileNetV2 (pre-trained on ImageNet). The weights of the base model are initially frozen.
*   **Custom Head:**
    *   Input Layer (matching specified resolution, e.g., 224x224x3 or 100x100x3).
    *   Data Augmentation Layer (random flips, rotations, zoom).
    *   Preprocessing Layer (rescales pixel values to [-1, 1] as expected by MobileNetV2).
    *   The frozen MobileNetV2 base.
    *   GlobalAveragePooling2D layer.
    *   Dropout layer for regularization.
    *   Dense output layer with `softmax` activation for multi-class classification (number of units equals the number of fruit classes).
*   **Optimizer:** Adam.
*   **Loss Function:** Categorical Crossentropy.

## 8. Key Technologies

*   **Python:** Core programming language.
*   **TensorFlow & Keras:** For building, training, and running the deep learning model.
*   **OpenCV (cv2):** For webcam access, image processing (resizing, color conversion), and displaying video feed with overlays.
*   **NumPy:** For numerical operations, especially array manipulation.
*   **Matplotlib:** For plotting training history (accuracy and loss).
*   **Glob:** For finding background image files.
*   **OS, Sys, Time, Shutil:** Standard Python libraries for file system operations, argument parsing, etc.

## 9. Potential Future Improvements

*   More sophisticated data augmentation techniques.
*   Allow fine-tuning of later layers in the MobileNetV2 base model after initial training.
*   Develop a more user-friendly GUI (e.g., using Tkinter, PyQt, or a web interface).
*   Experiment with other efficient model architectures (e.g., EfficientNet).
*   Implement object detection (drawing bounding boxes around fruits) instead of full-frame classification for scenarios with multiple fruits or cluttered backgrounds.
*   Add options for adjusting confidence thresholds for display. 