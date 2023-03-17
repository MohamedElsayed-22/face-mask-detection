# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array, load_img, normalize
from sklearn.model_selection import train_test_split
import numpy as np
import os

######### note the used dataset is relatively small that is why the training converges very fast.

# initialize the path of the dataset to be loaded
DIRECTORY = ".\dataset"
CLASSES = ["with_mask", "without_mask"]
BATCH_SIZE = 64
DIMENSION = 224
LEARNING_RATE = 10e-4
EPOCHS = 10
INPUT_SHAPE = (DIMENSION, DIMENSION, 3) 

dataset = []   
label = []
temp_label = 0

print("Loading and preprocessing the data...")
for class_type in CLASSES:
    path = os.path.join('.\dataset', class_type)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        dataset.append(np.array(image))
        label.append(temp_label)
    temp_label += 1

dataset = np.array(dataset)
label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)
X_train = normalize(X_train, axis=0)
X_test = normalize(X_test, axis=0)
print("Data loading and preprocessing are done.")


model = Sequential([
    Conv2D(32, (3, 3), activation='relu' ,input_shape=INPUT_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer = 'he_uniform'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation= 'relu', kernel_initializer = 'he_uniform'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),

])

optimization = Adam(learning_rate = LEARNING_RATE, decay = LEARNING_RATE / 100)
model.compile(loss="binary_crossentropy", optimizer = optimization, metrics=["accuracy"])

# print(model.summary())    

print("Training starts...")
history = model.fit(X_train, 
                         y_train, 
                         batch_size = BATCH_SIZE, 
                         verbose = 1, 
                         epochs = 10,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )

print("Training done.")

print("Saving the trained model...")
model.save("face_mask_recognizer.model", save_format="h5")
print("Model saving is done.")

