# Adjust data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Re-compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Re-train the model with less aggressive augmentation
augmented_history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                              epochs=5, validation_data=(test_images, test_labels))

# Evaluate and compare
augmented_accuracy = augmented_history.history['val_accuracy'][-1]
print(f"Adjusted Augmented Accuracy: {augmented_accuracy:.4f}")