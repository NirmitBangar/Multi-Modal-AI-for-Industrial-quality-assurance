"""
Week 3: CNN for Image-Based Defect Detection (Keras Implementation)
====================================================================
This file builds a Convolutional Neural Network for visual quality inspection.

We use CIFAR-10 as a proxy for a defect detection dataset, as it:
  - Is freely available and loads via Keras in one line
  - Has similar image structure (32×32 RGB) to cropped defect patches
  - Allows clear demonstration of all CNN concepts

In the final project, this will be replaced with MVTec AD:
  the industry-standard benchmark for industrial defect detection.

CNN Architecture (VGG-inspired):
  Input (32×32×3)
  → Conv Block 1: [Conv(32,3×3) → BN → ReLU] × 2 → MaxPool
  → Conv Block 2: [Conv(64,3×3) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)
  → Conv Block 3: [Conv(128,3×3) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)
  → Flatten → FC(256) → Dropout(0.5) → Output(10, Softmax)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# =============================================================================

print("Loading CIFAR-10 dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training set: {X_train.shape} | dtype: {X_train.dtype}")
print(f"Test set:     {X_test.shape}")

# Normalize pixel values to [0, 1]
# Why: gradient descent works poorly when inputs are in [0, 255]
# (the scale of pixel values dwarfs the scale of weights, causing slow convergence)
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# Convert labels to one-hot encoding for categorical cross-entropy
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test, 10)

# Create validation split from training data
val_size = 5000
X_val, y_val_cat = X_train[-val_size:], y_train_cat[-val_size:]
X_train, y_train_cat = X_train[:-val_size], y_train_cat[:-val_size]

print(f"\nAfter split:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# =============================================================================
# SECTION 2: DATA AUGMENTATION
# =============================================================================
# Data augmentation is critical in visual inspection:
# - We artificially multiply the training set by creating modified versions
# - The network learns that orientation and brightness don't change defect identity
# - Each epoch, the network sees a slightly different version of each image
#
# For industrial QA, augmentations would also include:
# - Elastic distortions (simulate deformation)
# - Salt-and-pepper noise (simulate camera noise)
# - Random crops (simulate varying defect position)

datagen = ImageDataGenerator(
    horizontal_flip=True,       # Product defects can appear on either side
    rotation_range=15,          # Small rotations (images are mostly upright)
    width_shift_range=0.1,      # Translate up to 10% horizontally
    height_shift_range=0.1,     # Translate up to 10% vertically
    zoom_range=0.1,             # Zoom in/out up to 10%
    brightness_range=[0.8, 1.2], # Simulate lighting variation on factory floor
    fill_mode='nearest',         # Fill empty pixels by nearest neighbour
)
datagen.fit(X_train)


# =============================================================================
# SECTION 3: CNN ARCHITECTURE
# =============================================================================

def build_cnn(input_shape=(32, 32, 3), n_classes=10) -> keras.Model:
    """
    VGG-inspired CNN with Batch Normalization and Dropout.

    Design choices explained:
    - Progressive filter doubling (32→64→128): early layers detect simple patterns
      (edges, corners); deeper layers combine these into complex shapes
    - BatchNorm after Conv: normalizes activations, allows higher learning rates,
      acts as mild regularizer, speeds convergence
    - MaxPooling(2,2): downsamples spatial dims by 2x, introduces translation invariance
    - Dropout: randomly zeros activations during training → prevents co-adaptation
      of neurons → reduces overfitting. Different rates: 0.25 in conv blocks (lighter),
      0.5 in FC (heavier, since FC layers have more parameters)
    - L2 regularization on FC weights: penalizes large weights in the classification head
    """
    model = keras.Sequential([
        # --- Convolutional Block 1: Detect low-level features (edges, textures) ---
        layers.Conv2D(32, (3,3), padding='same', input_shape=input_shape,
                      kernel_initializer='he_normal',
                      name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal',
                      name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),                  # 32×32 → 16×16

        # --- Convolutional Block 2: Mid-level features (corners, blob shapes) ---
        layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal',
                      name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal',
                      name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),                  # 16×16 → 8×8
        layers.Dropout(0.25),

        # --- Convolutional Block 3: High-level features (object parts) ---
        layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal',
                      name='conv3_1'),
        layers.BatchNormalization(name='bn3_1'),
        layers.Activation('relu'),
        layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal',
                      name='conv3_2'),
        layers.BatchNormalization(name='bn3_2'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),                  # 8×8 → 4×4
        layers.Dropout(0.25),

        # --- Classification Head ---
        layers.Flatten(),                             # 4×4×128 = 2048 units
        layers.Dense(256, kernel_initializer='he_normal',
                     kernel_regularizer=keras.regularizers.l2(0.001),
                     name='fc1'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),                          # Heavy dropout in FC
        layers.Dense(n_classes, activation='softmax', name='output'),
    ])
    return model


model = build_cnn()
model.summary()

# Count parameters — demonstrates weight sharing efficiency of CNNs
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")
print(f"Parameters if fully connected (32×32×3 → 10): {32*32*3 * 10:,}")
# CNN has far fewer parameters due to weight sharing in conv layers


# =============================================================================
# SECTION 4: TRAINING CONFIGURATION
# =============================================================================

# Adam optimizer: combines momentum (SGD with momentum) and RMSProp (adaptive LR)
# Adam is the de facto standard for most deep learning tasks
# lr=0.001 is Adam's well-tuned default

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',    # Multi-class cross-entropy
    metrics=['accuracy',
             keras.metrics.AUC(name='auc')]   # AUC is more informative than accuracy
)

# Learning Rate Reduction on Plateau:
# If val_loss doesn't improve for 5 epochs, reduce LR by factor of 0.5
# This is a standard trick — high LR during early training (fast descent),
# lower LR when close to minimum (precise convergence)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Early Stopping: Halt training when val_loss hasn't improved for 15 epochs
# restore_best_weights=True: revert to the best checkpoint automatically
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)

# Model Checkpoint: Save best model to disk
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_cnn_model.h5', monitor='val_loss', save_best_only=True, verbose=0
)


# =============================================================================
# SECTION 5: TRAINING
# =============================================================================

BATCH_SIZE = 64    # Mini-batch size — standard for CIFAR-scale models
N_EPOCHS   = 80    # Max epochs (early stopping will likely trigger earlier)

print("\n--- Training CNN ---")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=N_EPOCHS,
    validation_data=(X_val, y_val_cat),
    callbacks=[lr_scheduler, early_stop, checkpoint],
    verbose=1
)


# =============================================================================
# SECTION 6: EVALUATION & VISUALIZATION
# =============================================================================

# Test set evaluation
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n--- Test Set Results ---")
print(f"Loss:     {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"AUC:      {test_auc:.4f}")

# Per-class breakdown
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = y_test.ravel()

print("\nPer-Class Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# --- PLOTS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CNN Training Results — CIFAR-10 (QA Proxy)', fontsize=14, fontweight='bold')

# Training curves
ax = axes[0, 0]
ax.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
ax.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
ax.set_title('Accuracy over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax.set_title('Loss over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', ax=ax, cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Normalized Confusion Matrix')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Sample predictions
ax = axes[1, 1]
ax.axis('off')
n_show = 6
indices = np.random.choice(len(X_test), n_show, replace=False)
for j, idx in enumerate(indices):
    ax_img = fig.add_subplot(2, n_show, n_show + j + 1)
    ax_img.imshow(X_test[idx])
    predicted = CLASS_NAMES[y_pred[idx]]
    actual    = CLASS_NAMES[y_true[idx]]
    color = 'green' if predicted == actual else 'red'
    ax_img.set_title(f'P:{predicted}\nA:{actual}', fontsize=6, color=color)
    ax_img.axis('off')

plt.tight_layout()
plt.savefig('cnn_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n[Note] CIFAR-10 is used as a visual classification proxy.")
print("In the final project, this CNN will be fine-tuned on MVTec AD —")
print("an industrial defect detection benchmark with 15 object categories.")
print("Transfer learning from a CIFAR-trained backbone reduces required")
print("labelled defect images from thousands to potentially hundreds.")
