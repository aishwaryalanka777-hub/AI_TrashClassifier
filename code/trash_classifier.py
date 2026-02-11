from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing import image

# Load Dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    '../dataset',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '../dataset',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training starting...🔥")

model.fit(train_generator, validation_data=validation_generator, epochs=20)
model.save("trash_model.h5")
print("Model saved successfully!")
print("Training finished ✅")
print("Loading test image...")

img = image.load_img('../dataset/test.jpg', target_size=(64,64))
img_array = image.img_to_array(img) / 255
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

classes = list(train_generator.class_indices.keys())

print("Prediction probabilities:", prediction)
print("Predicted class:", classes[np.argmax(prediction)])
