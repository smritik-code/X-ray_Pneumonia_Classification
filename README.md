# X-ray Pneumonia Classification with Grad-CAM

This project implements a **binary classifier** to detect **pneumonia from chest X-ray images** using a **fine-tuned ResNet-50**. Grad-CAM is integrated to provide interpretable visualizations of the regions influencing the model’s predictions.

## Highlights

* **Binary Classification**: Normal vs Pneumonia.
* **Fine-tuned ResNet-50**:
  * Final layer trained for **9 epochs**.
  * Layer4 fine-tuned for **3 epochs**.
* **Grad-CAM Visualization**: Highlights regions contributing to model predictions.
* **Robust Preprocessing**: CLAHE contrast enhancement, resizing to 224×224, and masking non-thoracic regions to focus the model on lungs.
* **Data Augmentation**: Rotation and translation applied to 30% of training images for robustness.


## Model Performance

| Metric        | Training                              | Validation |
| ------------- | ------------------------------------- | ---------- |
| Loss          | 0.0623                                | 0.0949     |
| Accuracy      | 0.9815                                | 0.9554     |
| Epochs        | 9 (final layer), 3 (fine-tune layer4) | -          |                    | -          |


## Methodology & Workflow

1. **Preprocessing & Masking**:

   * Apply thoracic mask to exclude irrelevant areas outside lungs.
   * Convert to grayscale → CLAHE for contrast → convert back to RGB → resize → normalize.

2. **Data Augmentation**:

   * 30% of training images rotated and translated.

3. **Model Architecture**:

   * ResNet-50 pretrained on ImageNet.
   * Last fully connected layer trained for 9 epochs.
   * Layer4 fine-tuned for 3 epochs.

4. **Prediction & Interpretability**:

   * Softmax output used with adjustable threshold for classification.
   * Grad-CAM overlays generated to visualize regions influencing predictions.


