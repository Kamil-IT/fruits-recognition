from keras.preprocessing.image import ImageDataGenerator

from generate_model.properties import TRAIN_DIR, VAL_DIR


def import_images():
    training_datagen = ImageDataGenerator()
    validation_datagen = ImageDataGenerator()

    train_gen = training_datagen.flow_from_directory(TRAIN_DIR,
                                                     target_size=(100, 100),
                                                     class_mode="categorical")  # multi-class classification.

    val_gen = validation_datagen.flow_from_directory(VAL_DIR,
                                                     target_size=(100, 100),
                                                     class_mode="categorical")  # multi-class classification.

    print(train_gen.class_indices)
    return train_gen, val_gen