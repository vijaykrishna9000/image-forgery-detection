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

    train_dir = 'data/images/train'
    val_dir = 'data/images/valid'

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary'
    )

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
