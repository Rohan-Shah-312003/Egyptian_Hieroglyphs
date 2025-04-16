from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D,GlobalMaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.models import Model

base = VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, 3))
base.trainable = False
for layer in base.layers:
    layer.trainable = False

x = base.output


x = GlobalMaxPooling2D()(x)

output = Dense(95, activation='softmax')(x)

model_vgg = Model(inputs=base.input, outputs=output)

model_vgg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy','Precision',"Recall"])

history_vgg=model_vgg.fit(train_gen,epochs=50,validation_data=valid_gen,
                                    callbacks=[
                                        learn_rate,
#                                         early_stop,
                                        checkpoint
                                              ]
                                   )

sns.set()
acc_vgg = history_vgg.history['accuracy']
val_acc_vgg = history_vgg.history['val_accuracy']
loss_vgg = history_vgg.history['loss']
val_loss_vgg = history_vgg.history['val_loss']
epochs_vgg= range(1, len(loss_vgg) + 1)
plt.plot(epochs_vgg, acc_vgg, color='green', label='Training Accuracy')
plt.plot(epochs_vgg, val_acc_vgg, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

y_ticks = [i * 0.1 for i in range(11)]  # Y-axis ticks from 0.0 to 1.0 with a step of 0.1
plt.yticks(y_ticks)

plt.legend()
plt.show()

# Evaluate the model on test data
results_test= model_vgg.evaluate(test_gen, steps=len(test_gen), verbose=1)
# results_test
test_loss_vgg, test_accuracy_vgg =results_test[0],results_test[1]
print(f'Test Accuracy: {test_accuracy_vgg}')
print(f'Test Loss: {test_loss_vgg}')

plt.figure()
plt.plot(epochs_vgg, loss_vgg, color='green', label='Training Loss')
plt.plot(epochs_vgg, val_loss_vgg, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

pd.DataFrame(history_vgg.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

results_train= model_vgg.evaluate(train_gen, steps=len(train_gen), verbose=1)
# results_valid
train_loss_vgg, train_accuracy_vgg =results_train[0],results_train[1]
print(f'Training Accuracy: {train_accuracy_vgg}')
print(f'Training Loss: {train_loss_vgg}')

# Evaluate the model on valid data
results_valid= model_vgg.evaluate(valid_gen, steps=len(valid_gen), verbose=1)
# results_valid
valid_loss_vgg, valid_accuracy_vgg =results_valid[0],results_valid[1]
print(f'Validation Accuracy: {valid_accuracy_vgg}')
print(f'Validation Loss: {valid_loss_vgg}')

# Evaluate the model on test data
results_test= model_vgg.evaluate(test_gen, steps=len(test_gen), verbose=1)
# results_test
test_loss_vgg, test_accuracy_vgg =results_test[0],results_test[1]
print(f'Test Accuracy: {test_accuracy_vgg}')
print(f'Test Loss: {test_loss_vgg}')

Evaluate_model_2(model_vgg, train_gen, valid_gen, test_gen)

Final_Report=[]
Final_Report.append(Evaluate_model_2(model_vgg, train_gen, valid_gen, test_gen))
Final_Report

models_list=[
#              "CNN",
#              "DenseNet",
             "VGG16",
#              "EfficientNetB0"
            ]

Models_Scores=pd.DataFrame(Final_Report,index=models_list,columns=["Train Loss","Test Loss",
                                                                   "Train Accuracy","Test Accuracy",
                                                                   "Train Precision","Test Precision",
                                                                   "Train Recall","Test Recall"])
Models_Scores

Models_Scores.to_csv(model_path+"ModelsScoresVGG16v.csv")

# model=keras.models.load_model(model_path+'model_cp.keras')
# model.save(model_path+'model_vgg16_1.h5')

model_cnn = keras.models.Sequential()

model_cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=keras.layers.ReLU(), input_shape=(224, 224, 3)))
model_cnn.add(keras.layers.BatchNormalization())

model_cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.BatchNormalization())
model_cnn.add(keras.layers.MaxPool2D(pool_size=2))

model_cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.BatchNormalization())
model_cnn.add(keras.layers.MaxPool2D(pool_size=2))

model_cnn.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.BatchNormalization())
model_cnn.add(keras.layers.MaxPool2D(pool_size=2))


model_cnn.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.BatchNormalization())
model_cnn.add(keras.layers.MaxPool2D(pool_size=2))

model_cnn.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.BatchNormalization())
model_cnn.add(keras.layers.MaxPool2D(pool_size=4))


model_cnn.add(keras.layers.Flatten())
model_cnn.add(keras.layers.Dense(256, activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.Dropout(0.1))
model_cnn.add(keras.layers.Dense(128, activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.Dropout(0.1))
model_cnn.add(keras.layers.Dense(128, activation=keras.layers.ReLU()))
model_cnn.add(keras.layers.Dense(95, activation="softmax", name="Output"))

model_cnn.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model_cnn.compile(loss="categorical_crossentropy", optimizer= optimizer, metrics=["accuracy",'Precision',"Recall"])
history_cnn = model_cnn.fit(train_gen,epochs= 80,
                    validation_data = valid_gen,
                    callbacks = [
#                         early_stop,
                        learn_rate,
                        checkpoint
                                ]
                    )

sns.set()
acc = history_cnn.history['accuracy']
val_acc = history_cnn.history['val_accuracy']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, color='green', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

pd.DataFrame(history_cnn.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

results_train_cnn= model_cnn.evaluate(train_gen, steps=len(train_gen), verbose=1)
# results_valid
train_loss_cnn, train_accuracy_cnn =results_train_cnn[0],results_train_cnn[1]
print(f'Training Accuracy: {train_accuracy_cnn}')
print(f'Training Loss: {train_loss_cnn}')

# Evaluate the model on valid data
results_valid_cnn= model_cnn.evaluate(valid_gen, steps=len(valid_gen), verbose=1)
# results_valid
valid_loss_cnn, valid_accuracy_cnn =results_valid_cnn[0],results_valid_cnn[1]
print(f'Validation Accuracy: {valid_accuracy_cnn}')
print(f'Validation Loss: {valid_loss_cnn}')

# Evaluate the model on test data
results_test_cnn= model_cnn.evaluate(test_gen, steps=len(test_gen), verbose=1)
# results_test
test_loss_cnn, test_accuracy_cnn =results_test_cnn[0],results_test_cnn[1]
print(f'Test Accuracy: {test_accuracy_cnn}')
print(f'Test Loss: {test_loss_cnn}')
Evaluate_model_2(model_cnn, train_gen, valid_gen, test_gen)
Final_Report.append(Evaluate_model_2(model_cnn, train_gen, valid_gen, test_gen))
models_list=[
#              "DenseNet",
             "VGG16",
             "CNN",
#              "EfficientNetB0"
            ]

Models_Scores=pd.DataFrame(Final_Report,index=models_list,columns=["Train Loss","Test Loss",
                                                                   "Train Accuracy","Test Accuracy",
                                                                   "Train Precision","Test Precision",
                                                                   "Train Recall","Test Recall"])
Models_Scores
Models_Scores.to_csv(model_path+"ModelsScoresVGG16v_CNNv.csv")
model=keras.models.load_model(model_path+'model_cp.keras',custom_objects={'ReLU': keras.layers.ReLU})
model.save(model_path+'model_cnn_1.h5')
def predict_single_image_from_csv(model, data_csv, image_index, class_names,data_dir):

    # Get the image path for the specified index
    img_path = data_csv.iloc[image_index]['filename']  # Adjust 'image_path' to your column name

    # Read and preprocess the image
    img = cv2.imread(data_dir+"/"+img_path)
    img = cv2.resize(img, (HEIGHT, WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype("float32") / 255.0  # Normalize the image

    # Make a prediction
    prediction = np.argmax(model.predict(img.reshape(1, HEIGHT, WIDTH, 3), verbose=1), axis=-1)[0]

    # Get the actual label from the DataFrame
    actual = data_csv.iloc[image_index]['class']  # Adjust 'label' to your column name

    # Display the image and prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Actual: {actual}\nPredicted: {class_names[prediction]}")
    plt.axis('off')  # Optional: turn off axes
    plt.show()

image_index = 240
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)
image_index = 242
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)
image_index = 244
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)
image_index = 246
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)
image_index = 248
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)
image_index = 250
predict_single_image_from_csv(model_cnn, test_csv, image_index,classes_train,test_dir)