from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image

if __name__ == '__main__':
    model = train_model()
    evaluate_model(model)
    img_path = 'data/images/test/fake/0001.jpg'
    pred = predict_image(model, img_path)
    print(f"Prediction: {'FAKE' if pred == 1 else 'REAL'}")

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model():
    image_size = 224
    batch_size = 32
    epochs = 10
    data_dir = 'data/images/'  

    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(image_size, image_size),
        batch_size=batch_size, class_mode='binary', subset='training')
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(image_size, image_size),
        batch_size=batch_size, class_mode='binary', subset='validation')

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('models/best_model.h5', save_best_only=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    return model

from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    image_size = 224
    data_dir = 'data/images/'

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(image_size, image_size),
        batch_size=32, class_mode='binary', subset='validation', shuffle=False)

    predictions = (model.predict(val_gen) > 0.5).astype(int)
    print(classification_report(val_gen.classes, predictions))


import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = (model.predict(img_array) > 0.5).astype(int)[0][0]
    return prediction
