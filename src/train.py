from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Preprocessing + Augmentation (TRAIN)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/Test (NO augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

print("✅ Data preprocessing and loading done!")



from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base model
base_model.trainable = False

# Custom layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

print("✅ Model training completed!")


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Predictions
predictions = model.predict(test_data)
y_pred = (predictions > 0.5).astype(int)

# True labels
y_true = test_data.classes

# Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))