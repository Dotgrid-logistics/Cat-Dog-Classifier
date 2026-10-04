  import numpy as np
  import streamlit as st
  import tensorflow as tf
  from PIL import Image

  IMG_SIZE = (160, 160)
  class_names = ['cats', 'dogs']   # paste your printed class_names, same order

  @st.cache_resource
  def load_model():
      return tf.keras.models.load_model('cat_dog_model.keras', compile=False)

  model = load_model()

  st.title("Cat vs Dog Classifier")
  file = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png"])

  if file:
      img = Image.open(file).convert("RGB")
      st.image(img, use_container_width=True)
      arr = np.array(img.resize(IMG_SIZE), dtype="float32")[np.newaxis]
      p = float(tf.sigmoid(model.predict(arr, verbose=0))[0][0])
      label = class_names[1] if p > 0.5 else class_names[0]
      conf = p if p > 0.5 else 1 - p
      st.subheader(f"{label} ({conf:.1%} confident)")
