import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pathlib import Path
import matplotlib.pyplot as plt

# Defining Properties
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 30
FINE_TUNE_EPOCHS = 20
NUM_CLASSES = 7  # Types of cancer
INITIAL_LEARNING_RATE = 1e-5
FINE_TUNE_LEARNING_RATE = 1e-6

# Filepaths
MAIN_FOLDER = Path("C:/Users/Admin/Desktop/Skin-cancer/data")
TRAIN_FOLDER = MAIN_FOLDER / "train"
TEST_FOLDER = MAIN_FOLDER / "test"
VALIDATION_FOLDER = MAIN_FOLDER / "validation"

def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Model config
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # How the final model should look
    model = Model(inputs=base_model.input, outputs=outputs)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_FOLDER,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = test_val_datagen.flow_from_directory(
        VALIDATION_FOLDER,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_val_datagen.flow_from_directory(
        TEST_FOLDER,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, validation_generator, test_generator

def plot_training_history(history, fine_tune_history=None):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if fine_tune_history:
        plt.plot(range(len(history.history['accuracy']), len(history.history['accuracy']) + len(fine_tune_history.history['accuracy'])),
                 fine_tune_history.history['accuracy'])
        plt.plot(range(len(history.history['val_accuracy']), len(history.history['val_accuracy']) + len(fine_tune_history.history['val_accuracy'])),
                 fine_tune_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation', 'Train Fine-tune', 'Validation Fine-tune'], loc='lower right')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if fine_tune_history:
        plt.plot(range(len(history.history['loss']), len(history.history['loss']) + len(fine_tune_history.history['loss'])),
                 fine_tune_history.history['loss'])
        plt.plot(range(len(history.history['val_loss']), len(history.history['val_loss']) + len(fine_tune_history.history['val_loss'])),
                 fine_tune_history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation', 'Train Fine-tune', 'Validation Fine-tune'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def unfreeze_model(model):
    for layer in model.layers[-20:]:
        layer.trainable = True

def train_model():
    model = create_model()
    model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    train_generator, validation_generator, test_generator = create_data_generators()

    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-7, verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[lr_reducer]
    )

    # Fine-tuning
    unfreeze_model(model)
    model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[lr_reducer]
    )

    # Ploting Graphs
    plot_training_history(history, history_fine)

    # Evaluation
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")

    model.save('ham10000_resnet50_model.keras')

if __name__ == "__main__":
    train_model()