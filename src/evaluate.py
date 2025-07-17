from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model):
    image_size = 224
    data_dir = 'data/images/'

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(image_size, image_size),
        batch_size=32, class_mode='binary', subset='validation', shuffle=False)

    predictions = (model.predict(val_gen) > 0.5).astype(int)
    print(classification_report(val_gen.classes, predictions))
