import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Definir caminhos para os diretórios de treino e validação
train_dir = 'data/train'
validation_dir = 'data/validation'

# Criar geradores de dados de treino e validação, com normalização
train_datagen = ImageDataGenerator(rescale=1./255)  # Normaliza os valores de pixel para o intervalo [0, 1]
validation_datagen = ImageDataGenerator(rescale=1./255)

# Carregar imagens do diretório e aplicar redimensionamento e agrupamento por classes (cachorros e gatos)
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Diretório de treino
    target_size=(150, 150),  # Redimensiona as imagens para 150x150 pixels
    batch_size=20,
    class_mode='binary'  # Indica que é uma classificação binária (gato ou cachorro)
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # Diretório de validação
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = models.Sequential([
    # Camada de convolução para detectar características visuais
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),  # Pooling para reduzir a dimensão
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flattening para achatar a matriz 3D em um vetor 1D
    layers.Flatten(),
    
    # Camadas densas totalmente conectadas
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Saída binária: gato ou cachorro
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Quantas vezes passar pelos dados por época
    epochs=15,  # Quantas vezes todo o conjunto de treino é usado
    validation_data=validation_generator,
    validation_steps=50  # Quantos batches usar para validação
)

model.evaluate(validation_generator)
model.save('model/cats_vs_dogs_model.h5')
