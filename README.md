# Autonomous Robot Bottle Detection Using MobileNetV2

## Overview
This repository trains a binary classifier (bottle vs. no bottle) by fine-tuning MobileNetV2 on a custom dataset.


## Key Steps

1. **Imports & Setup**  
   - Common libraries: `numpy`, `pandas`, `matplotlib`, `tensorflow.keras`, `keras`, `sklearn`.
   - `MobileNetV2` backbone loaded with pre-trained weights (no top).

2. **Data Loading & Labels**  
   - Collect filepaths under `./data/Bottles` and `./data/Non_Bottles`.
   - Assign label `'b'` for bottle images, `'nb'` for no-bottle images.
   - Total samples: 52,577 (45,272 bottles + 7,305 non-bottles).

3. **Train/Validation/Test Split**  
   - 80% of data for training (further split 80/20 into train/validation), 20% for testing.
   - Create three Pandas DataFrames with columns `['Images', 'target']` for each split.

4. **Data Generators**  
   - **Training**: `ImageDataGenerator` with rescaling and augmentations (flip, rotation, shift, zoom).  
   - **Validation/Test**: `ImageDataGenerator` with only rescaling.
   - Use `flow_from_dataframe` to feed images of size 224×224, binary labels (`b=1, nb=0`), batch_size=32.

5. **Model Definition**  
   - Load `MobileNetV2(include_top=False, weights=<local-.h5-file>, input_shape=(224,224,3))`.  
   - Append:  
     - `GlobalAveragePooling2D()`  
     - `Dense(1024, activation='relu')`  
     - `Dense(512, activation='relu')`  
     - `Dense(1, activation='sigmoid')`  
   - Freeze all base layers initially.

6. **Compilation & Checkpointing**  
   - Compile with `Adam(learning_rate=0.001)`, `binary_crossentropy`, `accuracy`.  
   - Use `ModelCheckpoint` to save the best model (monitor `val_loss`, mode=`min`).

7. **Training Phase 1 (Feature Extraction)**  
   - Train for 5 epochs on frozen backbone.  
   - Save training history to `models/history_initial.pkl`.

8. **Fine-Tuning Phase**  
   - Unfreeze last 30 layers of MobileNetV2.  
   - Recompile with lower `learning_rate=0.0001`.  
   - Train for 5 more epochs.  
   - Save fine-tuned history to `models/history_finetune.pkl`.

9. **Results & Plots**  
   - Combine both histories to plot overall training vs. validation loss/accuracy.  
   - Final validation accuracy reached ≈99.95% with extremely low loss.

10. **Evaluation & Inference**  
    - Load best model (`./models/best_model.h5`).  
    - Evaluate on validation/test generator:  
      ```
      loss, accuracy = best_model.evaluate(val_generator)
      ```
    - Example prediction function:
      ```python
      def predict_image(path):
          img = load_img(path, target_size=(224,224))
          arr = img_to_array(img) / 255.
          pred = best_model.predict(np.expand_dims(arr, axis=0))[0][0]
          return "no bottle" if pred > 0.5 else "bottle"
      ```
    - Tested on `./test/20250311_154953.jpg`, yielding “bottle.”

## Example Screenshots

![No Bottle Detection](no_bottle.jpg)  
*No Bottle Detection*

![Bottle Detection](bottle.jpg)  
*Bottle Detection*

## Requirements
- Python 3.6+  
- TensorFlow/Keras, OpenCV, NumPy, pandas, scikit-learn, matplotlib, h5py (for saving/loading models)

## Contact
For questions or collaboration, contact:  
**mariusc0023@gmail.com**

