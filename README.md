# Introduction
This notebook trains a simple CNN with about 5 million parameters from scratch and finally reached an averaged accuracy of 90% and a weighted accuracy of 89% in the validation set. It demonstrat the potential viability to further develop an applicable model based on the dataset, indicating a strong patterns the CNN can learn from. 

Further works may need be done to gain a better performance of classification:
- Try a more complex CNN model to get a better performance. In the meanwhile, increase the input size, which means less compression on original picture.
- Collect more data of bad performed class and fine tune on them to address the specific bad performance on several categories.
- Try transfer learning on current large CNN models with the dataset, which is possible to reach applicable performances.

# About the dataset 
The dataset is an extensive eye disease dataset containing original and augmented datasets of a variety of eye diseases including 10 categories: "Retinitis Pigmentosa, Retinal Detachment, Pterygium, Myopia, Macular Scar, Glaucoma, Disc Edema, Diabetic Retinopathy, Central Serous Chorioretinopathy, and Healthy eye image".

Dataset in Kaggle link: https://www.kaggle.com/datasets/ruhulaminsharif/eye-disease-image-dataset

The original dataset of color fundus images for the detection and classification of eye diseases come from this paper: (https://www.sciencedirect.com/science/article/pii/S2352340924009417)

# How to use
Load the model using tensorflow: 

```python
import tensorflow as tf
# This model is saved as a full model in tensorflow, including the model structure and weights
MODEL_PATH = '/final_model_eye_disease_afterfinetune.keras' 
model = tf.keras.models.load_model(MODEL_PATH)
model.compile(...)
history = model.fit(...)
```

# Look the notebook and get futher understand
