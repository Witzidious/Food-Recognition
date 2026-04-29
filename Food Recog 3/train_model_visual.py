import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import cbam as cb


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = r"dataset"

# Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_labels = train_ds.class_names
num_classes = len(class_labels)
print("Classes:", class_labels)

# Tiền xử lý
data_preprocessing = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomCrop(210, 210),
    tf.keras.layers.RandomRotation(0.1),
    # tf.keras.layers.RandomZoom(0.1),
     tf.keras.layers.RandomContrast(0.1),
    # tf.keras.layers.RandomBrightness(0.025),
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE)
])
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (data_preprocessing(x / 255.0), y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# CNN
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = Conv2D(32, (3,3), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64, (3,3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = cb.cbam(x)
x = MaxPooling2D(2,2)(x)
x = Dropout(0.3)(x)

x = Conv2D(128, (3,3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(128, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = cb.cbam(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(256, (3,3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3,3), padding='same')(x)
x = Activation('relu')(x)
x = cb.cbam(x)

x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

'''
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)
'''

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train model
H = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    # callbacks=[early_stop] 
)

y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

# Predict
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n Confusion Matrix \n")
print(cm)

# Classification Report
cr = classification_report(y_true, y_pred, target_names=class_labels)
print("\n Classification Report \n")
print(cr)

report_path = os.path.join("Food Recog 3", "report.txt")
with open(report_path, "w") as f:
    f.write("1. Classification Report\n")
    f.write(cr)
    f.write("\n2. Confusion Matrix\n")
    f.write(np.array2string(cm))

# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(H.history['accuracy'], label='Train Accuracy')
plt.plot(H.history['val_accuracy'], label='Val Accuracy')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
# Loss
plt.subplot(1, 2, 2)
plt.plot(H.history['loss'], label='Train Loss')
plt.plot(H.history['val_loss'], label='Val Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Lưu model
model.save(r"Food Recog 3\model\food_model.keras")
json.dump(class_labels, open(r"Food Recog 3\model\labels.json", "w"))

print("\nTraining model completed")
