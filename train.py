from image_process import (create_dataset, display_augmented_images, display_image, 
                           plot_loss_curves, make_confusion_matrix, pred_and_plot,
                           calculate_results, load_and_prep_image, generate_class_weights)

import numpy as np
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
import matplotlib.image as mpmig

# set the seed
keras.utils.set_random_seed(42)

# define image path
IMAGE_PATH = "brain-tumor-mri-images-44-classes"

# set some hyperparameters
BATCH_SIZE=32
IMAGE_WIDTH=228
IMAGE_HEIGHT=228

# create datasets
train_ds = create_dataset("train",
                image_path=IMAGE_PATH,  
                subset_type="training", 
                image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                batch_size=BATCH_SIZE,
                split_size=.2,
                seed=42)
valid_ds = create_dataset("validation",
                image_path=IMAGE_PATH,  
                subset_type="validation", 
                image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                batch_size=BATCH_SIZE,
                split_size=.2,
                seed=42
)
classes = train_ds.class_names
valid_classes = valid_ds.class_names

y_train= tf.concat([y for x, y in train_ds], axis=0)
y_true = np.concatenate([y for x, y in valid_ds], axis=0)

# add some normalization and data augmentation
normalization_layer = tf.keras.layers.Rescaling(1./255)

# apply norm and describing it as new dataset
train_data = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
valid_data = valid_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# apply prefetch to both dataset
train_data = train_data.prefetch(tf.data.AUTOTUNE)
valid_data = valid_data.prefetch(tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# augmentation
# data_augmentation = Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(.5),
#     layers.RandomZoom(.4)
# ])

#display_augmented_images(valid_ds, augmentation=data_augmentation, img_count=12)

# create model
inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
#x = data_augmentation(inputs)
x = Conv2D(128, kernel_size=3, activation="relu", padding="same")(inputs)
x = Conv2D(128, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(2)(x)
x = layers.Dropout(.2)(x)
x = Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = Conv2D(64, kernel_size=3, activation="relu", padding="same")(x)
x = MaxPooling2D(2)(x)

x = Flatten(name="Flatten")(x)
outputs = Dense(44, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# generate class weights
class_weights = generate_class_weights(y_train.numpy())
print(class_weights)

###### callbacks ########
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

checkpoint_path = "checkpoints/cp.cpkt"
model_checkpoint = ModelCheckpoint(checkpoint_path,
                                    monitor="val_accuracy",
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1)

mixed_precision.set_global_policy(policy="mixed_float16")
print(mixed_precision.global_policy())

# compile model
model.compile(
    optimizer=Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history_ds = model.fit(train_ds, validation_data=valid_ds,
                       batch_size=BATCH_SIZE,
                       epochs=50,
                       callbacks=[early_stopping, model_checkpoint],
                       class_weight=class_weights)

y_pred = model.predict(valid_ds, verbose=1)
y_pred = np.argmax(y_pred, axis=1)

plot_loss_curves(history_ds)
pred_and_plot(model, "Astrocitoma T2.jpeg", classes, scale=False)
calculate_results(y_true, y_pred)
make_confusion_matrix(y_true, y_pred, figsize=(30,30), text_size=20, classes=valid_classes, savefig=True)