import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# from tensorflow.keras import layers, models
import numpy as np

model = tf.keras.models.load_model('model/cats_vs_dogs_model.h5')

# Carregar e processar a imagem
img_path = 'data/test/image_2.png'  # Altere para o caminho da sua imagem
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0  # Normaliza
img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão extra

# Prever a classe
predictions = model.predict(img_array)
class_label = 'Cachorro' if predictions[0] > 0.5 else 'Gato'
print(f'A imagem é: {class_label}')
