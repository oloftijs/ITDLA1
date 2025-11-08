#a1_task_2_final %%

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras


# %% [markdown]
# we load the images and labels into a np array

# %%
pixel_size = 150
data_folder = f"A1_data_{pixel_size}"
images_path = os.path.join(data_folder, "images.npy")
images = np.load(images_path)
images = images.astype('float32') / 255.0  # Normalize images
labels_path = os.path.join(data_folder, "labels.npy")
labels = np.load(labels_path)
#set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## Classification
# For our first experiment, we will treat the problem as a classification model
# We will run this experiment with 12, 72, and 720 classes

# %% [markdown]
# Function that allows us to make N_classes (note: it does not work for less than 12 classes)

# %%

def get_cat_labels(labels, N_classes):
    new_labels = []
    for label in labels:
        label = label[0]* N_classes/12+ int(label[1]* N_classes/(12*60))

        new_labels.append(int(label))
    return np.array(new_labels)


# %% [markdown]
# We then split the data into training, validation, and test sets. The sklearn train_test_split method shuffles the data by default

# %%

X_train_full, X_test,y_train_full, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=35
)
X_train, X_valid,y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=1/9, random_state=35
) # 1/9 x 0.9 = 0.1.


# %% [markdown]
# We define a common sense Error. This calculates how far of the prediction was

# %%
def make_common_sense_loss(num_classes=24.0):
    def common_sense_loss(y_true, y_pred):
        y_pred_class = tf.argmax(y_pred, axis=1)
        y_true_float = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
        y_pred_float = tf.cast(y_pred_class, dtype=tf.float32)
        diff = tf.abs(y_true_float - y_pred_float)
        cyclical_diff = tf.minimum(diff, num_classes- diff)
        return tf.reduce_mean(cyclical_diff * 720 /num_classes)
    return common_sense_loss

# %% [markdown]
# We define our model architecture
# We use a learning rate reducer to reduce our learning rate when no improvement is found after 5 epochs, stop our training if there is no improvement for 7 epochs, and use a csv logger to store how our losses develop during training.
# 

# %%
def create_model(num_classes=24):
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # halve the learning rate if there is no improvement
        patience=5,          # Wait 5 epochs with no improvement before reducing
        min_lr=1e-6          # Set a minimum learning rate at 1e-6
    )
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,          # Wait 7 epochs for improvement before stopping
        restore_best_weights=True  # Automatically restore the weights from the best epoch
    )
    csv_logger = keras.callbacks.CSVLogger(
    filename=f"classification_log{num_classes}_classes.csv",
    separator=",",
    append=True)  #makes sure results are appended to same file if training stops and is resumed
    model= keras.models.Sequential([
        keras.Input(shape=(pixel_size, pixel_size, 1)),
        # Block 1
        keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.1),
        # Block 2
        keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.1),
        # Block 3
        keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.1),
        # Block 4
        keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.1),


        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="leaky_relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="leaky_relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(int(num_classes), activation="softmax")
    ])
    cse = make_common_sense_loss(num_classes=num_classes)
    model.compile(loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=[cse,"Accuracy"
            ],
    )
    return model, lr_scheduler, early_stopper, csv_logger


# %%

model = keras.models.load_model('classification_model_720_classes.keras',
custom_objects={'common_sense_loss': make_common_sense_loss(num_classes=720)})
train_labels = get_cat_labels(y_train,720)
valid_labels = get_cat_labels(y_valid,720)
_, lr_scheduler, early_stopper, csv_logger = create_model(num_classes=720)
model.fit(
        X_train, train_labels,
        epochs=70,
        batch_size=16,
        validation_data=(X_valid, valid_labels),
        callbacks=[lr_scheduler, early_stopper, csv_logger]
    )
for num_classes in [12,72,720]:
    model, lr_scheduler, early_stopper, csv_logger = create_model(num_classes=num_classes)
    train_labels = get_cat_labels(y_train,num_classes)
    valid_labels = get_cat_labels(y_valid,num_classes)
    test_labels = get_cat_labels(y_test,num_classes)
    history = model.fit(
        X_train, train_labels,
        epochs=50,
        batch_size=16,
        validation_data=(X_valid, valid_labels),
        callbacks=[lr_scheduler, early_stopper, csv_logger]
    )
    test_loss, test_cse, test_accuracy = model.evaluate(X_test, test_labels)
    model.save(f"classification_model_{num_classes}_classes.keras")
    print(f"{num_classes} Classes: \n Test Loss: {test_loss}, Test Common Sense Error: {test_cse}, Test Accuracy: {test_accuracy}")

# %% [markdown]
# Regression model. less dropout as this is less prone to overfitting

# %% [markdown]
# feat prep

# %%
import sklearn.preprocessing
labels_minutes = labels[:,0]*60 + labels[:,1]
labels_minutes = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(labels_minutes.reshape(-1,1)).flatten()
images = images.astype('float32') / 255.0
X_train_full, X_test,y_train_full, y_test = train_test_split(
    images, labels_minutes, test_size=0.1, random_state=35
)
X_train, X_valid,y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=1/9, random_state=35
)

# %%
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,          # halve the learning rate if there is no improvement
    patience=3,          # Wait 3 epochs with no improvement before reducing
    min_lr=1e-6          # Set a minimum learning rate at 1e-6
)
early_stopper = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,          # Wait 5 epochs for improvement before stopping
    restore_best_weights=True  # Automatically restore the weights from the best epoch
)
csv_logger = keras.callbacks.CSVLogger(
filename=f"regression_log.csv",
separator=",",
append=True)
model= keras.models.Sequential([
    keras.Input(shape=(pixel_size, pixel_size, 1)),
    #feature augmentation
    keras.layers.RandomRotation(factor=(12 / 360)),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    # Block 1
    keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    # keras.layers.Dropout(0.1),
    # Block 2
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    # keras.layers.Dropout(0.1),
    # Block 3
    keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    # keras.layers.Dropout(0.1),
    # Block 4
    keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),
    # keras.layers.Dropout(0.1),


    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="leaky_relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="leaky_relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation="linear")
])

model.compile(loss='mean_squared_error',
optimizer=keras.optimizers.Adam(learning_rate=0.001),
metrics=['MeanAbsoluteError'],
)

# %%
model.fit(
    X_train, y_train,
    epochs=50,
    # batch_size=16,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# %%
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

def common_sense_error_minutes(t_test, y_pred):
    error = abs(y_test - y_pred)
    error = np.minimum(error, 0.5 - error)
    return np.mean(error)

c_s_e = common_sense_error_minutes(y_test, y_pred)
print(f"Common Sense Error: {c_s_e*720:.2f}minutes")

# %% [markdown]
# Multi-head model 

# %%

inputs = keras.Input(shape=(pixel_size, pixel_size, 1), name="input_images")
x = keras.layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Dropout(0.15)(x)

x = keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.15)(x)

x = keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(2)(x)
x = keras.layers.Dropout(0.15)(x)


x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="leaky_relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = keras.layers.Dropout(0.3)(x)
hour_branch = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
hour_branch = keras.layers.Dropout(0.3)(hour_branch)
hour_output = keras.layers.Dense(2, activation="tanh", name="hour_output")(hour_branch)
minute_branch = keras.layers.Dense(128, activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)
                                   )(x)
minute_branch = keras.layers.Dropout(0.3)(minute_branch)
minute_output = keras.layers.Dense(2, activation="linear", name="minute_output")(minute_branch)

model = keras.Model(inputs=inputs, outputs=[hour_output, minute_output])
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    loss={
        "hour_output": "mean_squared_error",
        "minute_output": "mean_squared_error"
    },
    metrics={
        "hour_output": "mean_absolute_error",
        "minute_output": 'mean_absolute_error'
    }
)
for layer in model.layers:
    print(layer.name, '\n',layer.get_config())


# %%
#label prep for multi head model
X_train_full, X_test,y_train_full, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=35
)
X_train, X_valid,y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=1/9, random_state=35
) # 1/9 x 0.9 = 0.1.

y_hour_train = y_train[:,0] / 12.0
y_minute_train = y_train[:,1] / 60.0
y_hour_valid = y_valid[:,0] / 12.0
y_minute_valid = y_valid[:,1] / 60.0
y_hour_test = y_test[:,0] / 12.0
y_minute_test = y_test[:,1] / 60.0


# %%
model = create_multi_head_model(1)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # halve the learning rate if there is no improvement
        patience=3,          # Wait 3 epochs with no improvement before reducing
        min_lr=1e-6          # Set a minimum learning rate at 1e-6
    )
early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,          # Wait 5 epochs for improvement before stopping
        restore_best_weights=True  # Automatically restore the weights from the best epoch
    )
csv_logger = keras.callbacks.CSVLogger(
    filename="time_dual_head_training_log.csv",
    separator=",",
    append=True  #makes sure results are appended to same file if training stops and is resumed
)

model.fit(
    X_train, {"hour_output": y_hour_train, "minute_output": y_minute_train},
    epochs=60,
    loss_weights={"hour_output": 1, "minute_output": 0.5},
    validation_data=(X_valid, {"hour_output": y_hour_valid, "minute_output": y_minute_valid}),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
    )


# %%
def sin_cos_to_time(sin_cos,period):
    sin_component = sin_cos[:, 0]
    cos_component = sin_cos[:, 1]
    angles = np.arctan2(sin_component, cos_component)
    angle_positive = np.mod(angles, 2 * np.pi)
    time_values = (angle_positive / (2 * np.pi)) * period
    return np.round(time_values).astype(int)


# %%
model.save("time_dual_head_model_final2.keras")

model.evaluate(X_test, {"hour_output": y_hour_test, "minute_output": y_minute_test})
y_pred = model.predict(X_test)
#translate test and prediction to minutes
y_min_pred = y_pred[1].flatten() * 60.0 + y_pred[0].flatten() * 12.0 * 60.0
y_min_test = y_test[:,0] * 60  + y_test[:,1]
print(y_min_pred[:10])
print(y_min_test[:10])
error_in_minutes = y_min_test -y_min_pred
abs_error = np.abs(error_in_minutes)
cyclical_error = np.minimum(abs_error, 720 - abs_error)

print("Mean absolute error in minutes:", np.mean(cyclical_error))
#print accuracy
correct_predictions = np.sum(cyclical_error == 0)
total_predictions = len(cyclical_error)
accuracy = correct_predictions / total_predictions
print("Accuracy (exact time predictions):", accuracy)
#accuracy within 5 minutes
correct_within_5 = np.sum(cyclical_error <= 5)
accuracy_within_5 = correct_within_5 / total_predictions
print("Accuracy (predictions within 5 minutes):", accuracy_within_5)


# %% [markdown]
# label transformations

# %%
labels = np.load(labels_path)
#Separate hours and minutes
hours = labels[:, 0]
minutes = labels[:, 1]

#Transform hours into sine and cosine components
hour_sin = np.sin(2 * np.pi * hours / 12.0)
hour_cos = np.cos(2 * np.pi * hours / 12.0)
y_hour = np.stack([hour_sin, hour_cos], axis=1)
# Do the same for minutes
minute_sin = np.sin(2 * np.pi * minutes / 60.0)
minute_cos = np.cos(2 * np.pi * minutes / 60.0)
y_minute = np.stack([minute_sin, minute_cos], axis=1)

# Split into train/valid (80%) and test (20%)
X_train_full, X_test, \
y_hour_train_full, y_hour_test, \
y_minute_train_full, y_minute_test = train_test_split(
    images, y_hour, y_minute, test_size=0.2, random_state=35
)
X_train, X_valid, \
y_hour_train, y_hour_valid, \
y_minute_train, y_minute_valid = train_test_split(
    X_train_full, y_hour_train_full, y_minute_train_full,
    test_size=1/8, random_state=35
) # 1/8 x 0.8 = 0.1.


# %% [markdown]
# Build the model for the label transformations

# %%
model = create_multi_head_model(2)

# %%
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,          # halve the learning rate if there is no improvement
        patience=3,          # Wait 3 epochs with no improvement before reducing
        min_lr=1e-6          # Set a minimum learning rate at 1e-6
    )
early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,          # Wait 5 epochs for improvement before stopping
        restore_best_weights=True  # Automatically restore the weights from the best epoch
    )
csv_logger = keras.callbacks.CSVLogger(
    filename="time_dual_head_training_log.csv",
    separator=",",
    append=True  #makes sure results are appended to same file if training stops and is resumed
)

model.fit(
    X_train, {"hour_output": y_hour_train, "minute_output": y_minute_train},
    epochs=90,
    validation_data=(X_valid, {"hour_output": y_hour_valid, "minute_output": y_minute_valid}),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
    )


# %%
y_pred = model.predict(X_test)
pred_hours = sin_cos_to_time(y_pred[0], 12)
pred_minutes = sin_cos_to_time(y_pred[1], 60)
total_pred_minutes = pred_hours * 60 + pred_minutes

true_hours = sin_cos_to_time(y_hour_test, 12)
true_minutes = sin_cos_to_time(y_minute_test, 60)
print(pred_hours[10:], true_hours[10:])
total_true_minutes = true_hours * 60 + true_minutes
error_in_minutes = total_true_minutes - total_pred_minutes
abs_error = np.abs(error_in_minutes)
cyclical_error = np.minimum(abs_error, 720 - abs_error)

print("Mean absolute error in minutes:", np.mean(cyclical_error))
#print accuracy
correct_predictions = np.sum(cyclical_error == 0)
total_predictions = len(cyclical_error)
accuracy = correct_predictions / total_predictions
print("Accuracy (exact time predictions):", accuracy)
#accuracy within 5 minutes
correct_within_5 = np.sum(cyclical_error <= 5)
accuracy_within_5 = correct_within_5 / total_predictions
print("Accuracy (predictions within 5 minutes):", accuracy_within_5)


# %%
print(keras.__version__)

# %%
model.save("time_dual_head_model_final_transformers.keras")

# %%
import os
import matplotlib.pyplot as plt
folder_path = "csv_logs"
for filename in os.listdir(folder_path):
    if 'log' in filename:
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        #plot training and validation loss curves
        fig, ax1 =plt.subplots(figsize=(10,6))
        ax2 = None
        ax1.plot(df['loss'], label='Training Loss', color='blue')
        ax1.plot(df['val_loss'], label='Validation Loss', color='cyan')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        #if there is a column 'common_sense_loss' plot it too on a separate axis
        if 'common_sense_loss' in df.columns:
            ax2 = plt.twinx()
            ax2.plot(df['common_sense_loss'], label='Common Sense Error', color='orange')
            ax2.set_ylabel('Common Sense Error (minutes)')
        #if "dual_head" in filename plot hour and minute output losses & use logarithmic scale
        if 'dual_head' in filename:
            ax1.plot(df['hour_output_loss'], label='Hour Output Loss')
            ax1.plot(df['val_hour_output_loss'], label='Validation Hour Output Loss')
            ax1.plot(df['minute_output_loss'], label='Minute Output Loss')
            ax1.plot(df['val_minute_output_loss'], label='Validation Minute Output Loss')
            ax1.set_yscale('log')
        #create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = [], []
        if ax2 is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        fig.tight_layout()
        plt.savefig(f"{filename}_loss_curve.png")



