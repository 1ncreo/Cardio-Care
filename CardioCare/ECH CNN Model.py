import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths for training, testing, and prediction data
train_data_path = "D:/ECH cnn/data/train"
test_data_path = "D:/ECH cnn/data/test"
predict_data_path = "D:/ECH cnn/data/predict/Normal.png"

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data using the data generators
x_train = train_datagen.flow_from_directory(train_data_path, target_size=(64, 64), batch_size=32, class_mode="categorical")
x_test = test_datagen.flow_from_directory(test_data_path, target_size=(64, 64), batch_size=32, class_mode="categorical")

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation="relu"),
    Dense(units=6, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Determine the steps per epoch and validation steps
steps_per_epoch = len(x_train) 
validation_steps = len(x_test) 

# Train the model
model.fit(x_train, epochs=9, validation_data=x_test, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

# Predict the class of an ECG image
img = image.load_img(predict_data_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
y_pred = np.argmax(pred)

# Define the index of classes
index = [
    'left Bundle Branch block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contraction',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]

# Get the result
result = str(index[y_pred])
print(f"The predicted class for the ECG image is: {result}")

# Print a success message
print("Model training completed successfully.")