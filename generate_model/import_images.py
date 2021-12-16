from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from generate_model.properties import TRAIN_DIR, VAL_DIR, TEST_DIR


def import_images():
    training_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train = training_datagen.flow_from_directory(TRAIN_DIR,
                                                 target_size=(100, 100),
                                                 shuffle=True,
                                                 batch_size=34,
                                                 color_mode='rgb',
                                                 class_mode="categorical")  # multi-class classification.

    val = validation_datagen.flow_from_directory(VAL_DIR,
                                                 target_size=(100, 100),
                                                 shuffle=True,
                                                 color_mode='rgb',
                                                 class_mode="categorical")  # multi-class classification.

    test = test_datagen.flow_from_directory(TEST_DIR,
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=49,
                                            class_mode="categorical",  # multi-class classification.
                                            shuffle=False)

    print(train.class_indices)
    print(val.class_indices)
    print(test.class_indices)
    return train, val, test
