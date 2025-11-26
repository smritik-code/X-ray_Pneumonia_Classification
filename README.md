# X-ray Pneumonia Classification with Grad-CAM and Segmentation

This project implements a **binary classifier** to detect **pneumonia from chest X-ray images** using a **fine-tuned ResNet-50**. Grad-CAM is integrated to provide interpretable visualizations of the regions influencing the model’s predictions. Additionally, a **U-Net** model was trained to segment pneumonic regions, producing masks and bounding boxes over the lungs.

## Highlights

* **Binary Classification**: Normal vs Pneumonia.
* **Fine-tuned ResNet-50**:

  * Final layer trained for **9 epochs**.
  * Layer4 fine-tuned for **3 epochs**.
* **Grad-CAM Visualization**: Highlights regions contributing to model predictions.
* **U-Net Segmentation**: Produces masks for pneumonic regions, aiding interpretability.
* **Robust Preprocessing**: CLAHE contrast enhancement, resizing to 224×224, and masking non-thoracic regions to focus the model on lungs.
* **Data Augmentation**: Rotation and translation applied to 30% of training images for robustness.
* **Training / Validation Split**: U-Net trained with ~90% training images and ~10% for validation.

## Model Performance

| Metric              | Training                              | Validation |
| ------------------- | ------------------------------------- | ---------- |
| Classifier Loss     | 0.0623                                | 0.0949     |
| Classifier Accuracy | 0.9815                                | 0.9554     |
| Epochs              | 9 (final layer), 3 (fine-tune layer4) | -          |
| U-Net Train Loss    | 0.086                                 | 0.103      |
| U-Net Epochs        | 22                                    | -          |

## Methodology & Workflow

1. **Preprocessing & Masking**:

   * Apply thoracic mask to exclude irrelevant areas outside lungs.
   * Convert to grayscale → CLAHE for contrast → convert back to RGB → resize → normalize.

2. **Data Augmentation**:

   * 30% of training images rotated and translated to increase robustness.

3. **Model Architecture**:

   * ResNet-50 pretrained on ImageNet.
   * Last fully connected layer trained for 9 epochs.
   * Layer4 fine-tuned for 3 epochs.
   * **U-Net (ResNet-50 encoder)** trained for **22 epochs** to segment pneumonic regions.

4. **Prediction & Interpretability**:

   * Softmax output used with adjustable threshold for classification.
   * Grad-CAM overlays generated to visualize classifier attention.
   * U-Net outputs masks for pneumonia, which can be visualized as overlays or bounding boxes on original X-rays.


## Results

<img width="1903" height="978" alt="image" src="https://github.com/user-attachments/assets/77f9b1a7-9450-4bad-9575-72a607976b29" />
<img width="1294" height="967" alt="image" src="https://github.com/user-attachments/assets/5b8e3bcd-cd2d-422b-b7d2-3c0087e5d9ce" />
