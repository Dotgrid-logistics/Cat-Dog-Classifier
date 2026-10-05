# Cat vs Dog Image Classifier

A binary image classifier that tells cats and dogs apart, built with transfer learning on MobileNetV2 (TensorFlow/Keras), trained in Google Colab, and deployed as a live web app on Streamlit Community Cloud.

Live demo: https://ib3wnq9enrzmf6hm72hdrs.streamlit.app/

# Overview

This is Project 3: Teach a Computer to Recognize Something. The goal was to take a labeled image dataset, train a model on it, and turn the result into something anyone can try by uploading a photo.

Instead of training a network from scratch, the project reuses MobileNetV2, which was pretrained on ImageNet, and trains only a small classification head on top. This gives strong results with a small dataset, a few epochs, and a free Colab GPU.

# Results
Metric	Value
Test accuracy	fill in, e.g. 0.97
Best validation accuracy	fill in
Epochs	5
Input size	160 × 160

Add one sentence on what the numbers mean, e.g. "The model classifies unseen test images correctly about X% of the time."

# Dataset
Images are organised into train/ and test/ folders, each with cats/ and dogs/ subfolders.
The train/ folder is split 80/20 into training and validation sets (fixed seed 42 for reproducibility).
The test/ folder is kept separate and never used during training, so the final accuracy reflects unseen data.
Class names: ['cats', 'dogs']
Model Architecture
Data augmentation: random horizontal flip, rotation (±10%), zoom (±10%)
Preprocessing: MobileNetV2's preprocess_input
Base model: MobileNetV2 (ImageNet weights, frozen)
Head: Global Average Pooling → Dropout (0.2) → Dense(1) logit

# Training setup

Optimizer: Adam (learning rate 1e-3)
Loss: Binary cross-entropy (from_logits=True)
Metric: accuracy
Checkpoint: best model by validation accuracy
How Prediction Works

The final layer outputs a single logit. A sigmoid converts it to a probability:

probability > 0.5 → dog
probability ≤ 0.5 → cat

The confidence shown is the probability of the predicted class.

Project Structure
cat-dog-classifier/
├── cat_dog_model.keras   # trained model
├── app.py                # Streamlit app (if present in your repo)
├── requirements.txt      # dependencies
└── README.md

Adjust this to match the files actually in your repo.

# Run It Yourself

1. Clone the repo

bash
git clone https://github.com/Dotgrid-logistics/cat-dog-classifier.git
cd cat-dog-classifier

2. Install dependencies

bash
pip install -r requirements.txt

3. Run the app

bash
streamlit run app.py

Or use the model directly in Python

python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("cat_dog_model.keras")

img = tf.keras.utils.load_img("your_image.jpg", target_size=(160, 160))
arr = tf.keras.utils.img_to_array(img)[np.newaxis]
p = float(tf.sigmoid(model.predict(arr, verbose=0))[0][0])

print("dog" if p > 0.5 else "cat", f"({max(p, 1 - p):.1%} confident)")
Retraining

The full training notebook is in Google Colab. To retrain:

Upload your dataset (train/ and test/ folders) to Google Drive as archive.zip.
Run the notebook top to bottom with a GPU runtime.
Download the new cat_dog_model.keras and replace the one in this repo.
Limitations
Trained on two classes only. Any image (a rabbit, a car) will still be forced into "cat" or "dog".
Unusual cases, such as dogs that look like cats, can be misclassified.
The base model is frozen, so accuracy has room to improve with fine-tuning.
Next Steps
Fine-tune the top layers of MobileNetV2
Train for more epochs
Add an "unsure" threshold for low-confidence predictions
Add a confusion matrix and sample predictions to this report
Tech Stack

Python 3.12 · TensorFlow 2.20 · Keras · MobileNetV2 · Google Colab · Streamlit

Author

Praise, Computer Science student at TASUED. GitHub: Dotgrid-logistics
