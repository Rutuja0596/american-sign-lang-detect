import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

# Assuming you have your data organized in directories by class
train_data_dir = 'C:/Users/Rutuja/Desktop/sign-lang-detect/train_data'
val_data_dir = 'C:/Users/Rutuja/Desktop/sign-lang-detect/valid_data'

# Image data generators with augmentation
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

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Print number of samples to ensure data generators are correct
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")

# Ensure there are enough validation samples
if validation_generator.samples == 0:
    raise ValueError("No validation data found. Please ensure that the validation directory is correctly set up and contains images.")

# Define a more complex CNN model with Dropout layers to prevent overfitting
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Use EarlyStopping and ModelCheckpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('C:/Users/Rutuja/Desktop/sign-lang-detect/best_model.h5', save_best_only=True)

# Train the model
history = model.fit(
    train_generator, 
    epochs=10, 
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the model
model.save('C:/Users/Rutuja/Desktop/sign-lang-detect/sign_language_model.h5')

# Save the class labels to a file
class_labels = list(train_generator.class_indices.keys())
with open('C:/Users/Rutuja/Desktop/sign-lang-detect/class_labels.json', 'w') as f:
    json.dump(class_labels, f)
