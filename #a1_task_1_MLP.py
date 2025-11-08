#a1_task_1_MLP
# # MLP

# %% [markdown]
# initializations, activations, optimizers (and
# their hyperparameters), regularizations (L1, L2, Dropout, no Dropout). You may also experiment
# with changing the architecture of both networks: adding/removing layers, number of convolutional
# filters, their sizes, etc.

# %%
import tensorflow as tf
seed = 9+10
tf.random.set_seed(seed)

# %%
# Check number of available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
# Use TensorFlow's bundled Keras to ensure compatibility with GPUs
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler

random_state = 900
keras.utils.set_random_seed(random_state)

# Try to enable memory growth for all GPUs so TF doesn't reserve all GPU memory upfront
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('Enabled memory growth for GPUs:', gpus)
    except Exception as e:
        print('Could not set memory growth:', e)
else:
    print('No GPU devices found by TensorFlow')

# %% [markdown]
# ### Model 1
# 
# Below is our first attempt at building a neural network. We use the SGD optimizer.

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


# Choose hyperparameters
seed = 900
activation_function = 'relu6'
kernel_ini = keras.initializers.Orthogonal(gain = 1.0, seed = seed)
kernel_reg = keras.regularizers.l2(0.0001)
bias_ini = keras.initializers.Zeros()
shape = [28, 28]

# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=kernel_reg),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=6,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.SGD(learning_rate=5e-3),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

#evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %%
#evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
#0.8834999799728394

# %% [markdown]
# #### Application to CIFAR10
# We apply the model with chosen hyperparameters on the CIFAR10 set.

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


# Choose hyperparameters
seed = 900
activation_function = 'relu6'
kernel_ini = keras.initializers.Orthogonal(gain = 1.0, seed = seed)
bias_ini = keras.initializers.Zeros()
shape = [32,32,3]


# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=6,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.SGD(learning_rate=5e-3),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

#evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# ### Model 2

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


# Choose hyperparameters
seed = 900
activation_function = 'leaky_relu'

kernel_ini = "glorot_uniform"#keras.initializers.Orthogonal(gain = 1.0, seed = seed)
bias_ini = keras.initializers.Zeros()
shape = [28, 28]
learning_rate = 0.001


# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=kernel_reg),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
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


# Choose hyperparameters
seed = 900
activation_function = 'leaky_relu'

kernel_ini = "glorot_uniform"
bias_ini = keras.initializers.Zeros()
shape = [32,32,3]
learning_rate = 0.001


# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=kernel_reg),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# %% [markdown]
# 
# ### Model 3

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


# Choose hyperparameters
seed = 900
activation_function = 'relu6'

kernel_ini = keras.initializers.Orthogonal(gain = 1.0, seed = seed)
bias_ini = keras.initializers.Zeros()
shape = [28, 28]
learning_rate = 0.001


# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=kernel_reg),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#0.8765 

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

# Choose hyperparameters
seed = 900
activation_function = 'relu6'

kernel_ini = keras.initializers.Orthogonal(gain = 1.0, seed = seed)
bias_ini = keras.initializers.Zeros()
shape = [32,32,3]
learning_rate = 0.001


# Build model
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=shape),
    keras.layers.Dense(400, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(200, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, kernel_initializer = kernel_ini,
    activation=activation_function, kernel_regularizer=kernel_reg),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(50, kernel_initializer = kernel_ini,
    activation=activation_function),
    keras.layers.Dropout(0.05),
    keras.layers.Dense(10, kernel_initializer = kernel_ini,
    activation="softmax")
])
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True
)

model1.summary()


model1.compile(loss="sparse_categorical_crossentropy",
optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
metrics=["accuracy",
        #   tf.keras.metrics.Precision(), tf.keras.metrics.Recall()
          ])
model1.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_acc = model1.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


