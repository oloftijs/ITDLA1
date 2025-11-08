#a1_task_1_CNN %% [markdown]
# # CNN
# 
# Here we make a model 

# %%
# Check number of available GPUs
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
seed = 9+10
pixel_size=28
tf.random.set_seed(seed)

# %%
def create_model(num_classes=10, optimizer=keras.optimizers.Adam(learning_rate=0.0001), shape = (pixel_size, pixel_size, 1)):
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
    initializer = "glorot_uniform"# keras.initializers.Orthogonal(gain = 1.0, seed = seed)
    model= keras.models.Sequential([
        keras.Input(shape=shape),
        # Block 1
        keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.1),
        # Block 2
        keras.layers.Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.1),
        # Block 3
        keras.layers.Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.1),
        # Block 4
        keras.layers.Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer=initializer),
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
    model.compile(loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=["Accuracy"
            ],
    )
    return model, lr_scheduler, early_stopper, csv_logger


# %% [markdown]
# ## Model 1

# %%
# Import FashionMNIST data
fashion_mnist= keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, y_train_full.shape)

# Preprocess data
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

pixel_size = 28
model1, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

# %%
model1.summary()
model1.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# %%
# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# ### Application to CIFAR10

# %%
# Import CIFAR10 dataset
CIFAR10 = tf.keras.datasets.cifar10


# Preprocess data
(X_train_full, y_train_full), (X_test, y_test) = CIFAR10.load_data()
print(X_train_full.shape, y_train_full.shape)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

# %%
pixel_size = 32
modelA, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.Adam(learning_rate=0.0001),shape=(pixel_size, pixel_size, 3))

modelA.summary()
modelA.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# Evaluate the model on the test set
test_loss, test_acc = modelA.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# ## Model 2

# %%
# Import FashionMNIST data
fashion_mnist= keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, y_train_full.shape)

# Preprocess data
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

pixel_size = 28
model2, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.AdamW(learning_rate=0.0001))

# %%
model2.summary()
model2.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# %%
# Evaluate the model on the test set
test_loss, test_acc = model2.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# ### Application to CIFAR10

# %%
# Import CIFAR10 dataset
CIFAR10 = tf.keras.datasets.cifar10


# Preprocess data
(X_train_full, y_train_full), (X_test, y_test) = CIFAR10.load_data()
print(X_train_full.shape, y_train_full.shape)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

# %%
pixel_size = 32
modelB, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.AdamW(learning_rate=0.0001),shape=(pixel_size, pixel_size, 3))

modelB.summary()
modelB.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# Evaluate the model on the test set
test_loss, test_acc = modelB.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# ## Model 3

# %%
# Import FashionMNIST data
fashion_mnist= keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
print(X_train_full.shape, y_train_full.shape)

# Preprocess data
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

pixel_size = 28
model3, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.Nadam(learning_rate=0.0001))

# %%
model3.summary()
model3.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# %%
# Evaluate the model on the test set
test_loss, test_acc = model3.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# #### Application to CIFAR10

# %%
# Import CIFAR10 dataset
CIFAR10 = tf.keras.datasets.cifar10


# Preprocess data
(X_train_full, y_train_full), (X_test, y_test) = CIFAR10.load_data()
print(X_train_full.shape, y_train_full.shape)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_test = X_test / 255.0
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

# %%
pixel_size = 32
modelC, lr_scheduler, early_stopper, csv_logger  = create_model(optimizer=keras.optimizers.AdamW(learning_rate=0.0001),shape=(pixel_size, pixel_size, 3))

modelC.summary()
modelC.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_valid, y_valid),
    callbacks=[lr_scheduler, early_stopper, csv_logger]
)

# Evaluate the model on the test set
test_loss, test_acc = modelC.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


