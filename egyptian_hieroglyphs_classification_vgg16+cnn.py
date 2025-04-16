
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
alexandrepetit881234_egyptian_hieroglyphs_path = kagglehub.dataset_download('alexandrepetit881234/egyptian-hieroglyphs')

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

        print(dirname)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import kagglehub

path = kagglehub.dataset_download("alexandrepetit881234/egyptian-hieroglyphs")

print("Path to dataset files:", path)

import os
import cv2
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("done")

dataset_path="/kaggle/input/egyptian-hieroglyphs/"
train_path=dataset_path+"train"
train_csv=pd.read_csv(dataset_path+"train/_annotations.csv")
valid_path=dataset_path+"valid"
valid_csv=pd.read_csv(dataset_path+"valid/_annotations.csv")
test_path=dataset_path+"test"
test_csv=pd.read_csv(dataset_path+"test/_annotations.csv")

def load_bbox_from_csv(csv_path):
    """
    Load bounding box data from CSV file

    Parameters:
    csv_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: DataFrame containing image names and bounding box coordinates
    """
    try:
        df = pd.read_csv(csv_path)
        # Print the columns to help with debugging
        print("CSV columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def crop_images_with_bbox(image_dir, csv_path, output_dir):
    """
    Crop images based on bounding box coordinates from CSV.

    Parameters:
    image_dir (str): Directory containing the original images
    csv_path (str): Path to CSV file containing bounding box coordinates
    output_dir (str): Directory to save cropped images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    bbox_data = load_bbox_from_csv(csv_path)
    if bbox_data is None:
        return

    # Print first few rows to verify data
    print("\nFirst few rows of the CSV data:")
    print(bbox_data.head())

    # Process each row in the CSV
    for idx, row in bbox_data.iterrows():
        try:
            # Get image name and bbox coordinates using the correct column names
            image_name = row['filename']  # Changed from 'image_name'
            x_min = row['xmin']      # Changed from 'x_min'
            y_min = row['ymin']      # Changed from 'y_min'
            x_max = row['xmax']      # Changed from 'x_max'
            y_max = row['ymax']      # Changed from 'y_max'

            # Construct full image path
            image_path = os.path.join(image_dir, image_name)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_name} not found in {image_dir}")
                continue

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_name}")
                continue

            # Ensure coordinates are integers
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Crop image
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Save cropped image
            output_path = os.path.join(output_dir, f'{image_name}')
            cv2.imwrite(output_path, cropped_image)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            print(f"Row data: {row}")  # Added to help debug any issues
            continue

    print("Cropping completed!")

def process_dataset(image_dir, csv_path,folder):
    """
    Process the dataset images

    Parameters:
    image_dir (str): Directory containing images
    csv_path (str): Path to CSV file with bounding box coordinates
    """
    # Use /kaggle/working instead of the input directory

    output_dir = '/kaggle/working/cropped/'
    folders={
        0:"train",
        1:"valid",
        2:"test"
    }

    output_dir=output_dir+folders[folder]

    crop_images_with_bbox(image_dir, csv_path, output_dir)

# Example usage
if __name__ == "__main__":
    # For training set
    train_image_dir = train_path
    train_csv_path = train_path+'/_annotations.csv'
    process_dataset(train_image_dir, train_csv_path,0)

    # For validation set
    val_image_dir = valid_path
    val_csv_path = valid_path+'/_annotations.csv'
    process_dataset(val_image_dir, val_csv_path,1)

    # For testing set
    test_image_dir = test_path
    test_csv_path =test_path+'/_annotations.csv'
    process_dataset(test_image_dir, test_csv_path,2)

BATCH_SIZE=16
HEIGHT, WIDTH = 240,240

train_dir="/kaggle/working/cropped/train"
test_dir="/kaggle/working/cropped/test"
valid_dir="/kaggle/working/cropped/valid"

from sklearn.model_selection import train_test_split

# Split train and validation
train_df, valid_df = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv['class'],  # This ensures balanced split
    random_state=42
)

train_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    ).flow_from_dataframe(
                                      dataframe=train_df,
                                      directory= train_dir,
                                      x_col="filename",
                                      y_col="class",
                                      class_mode="categorical",
                                      shuffle=True,
                                      color_mode='rgb',
                                      batch_size=BATCH_SIZE,
                                      target_size=(WIDTH,HEIGHT),
                                      seed=0,
                          )
classes_train=list(train_gen.class_indices.keys())
plt.figure(figsize=(20,8))
for X_batch, y_batch in train_gen:
    for i in range(0,10):
        plt.subplot(2,5,i+1)
        plt.imshow(X_batch[i])
        plt.title(classes_train[np.where(y_batch[i]==1)[0][0]])
        plt.grid(None)
    plt.show()
    break

from collections import Counter

# Assuming you have train_gen set up using ImageDataGenerator
# This will give you a list of class indices from the training generator
class_indices = train_gen.classes  # Get class labels

# Count the occurrences of each class
class_counts = Counter(class_indices)

# Get class names and counts
class_names = list(class_counts.keys())
counts = list(class_counts.values())

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(class_names, counts, color='skyblue')
plt.title('Class Distribution in Training Data')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

valid_gen=ImageDataGenerator(
    rescale=1./255,
    ).flow_from_dataframe(
                            dataframe=valid_df,
                            directory=train_dir,
                            x_col="filename",
                            y_col="class",
                            class_mode="categorical",
                            shuffle=False,
                            color_mode='rgb',
                            batch_size=BATCH_SIZE,
                            target_size=(WIDTH, HEIGHT),
                            seed=0
                          )
classes_valid=list(valid_gen.class_indices.keys())
plt.figure(figsize=(10,6))
for X_batch, y_batch in valid_gen:
    for i in range(0,8):
        plt.subplot(2,4,i+1)
        plt.imshow(X_batch[i])
        plt.title(classes_valid[np.where(y_batch[i]==1)[0][0]])
        plt.grid(None)
    plt.show()
    break

from collections import Counter

# Assuming you have train_gen set up using ImageDataGenerator
# This will give you a list of class indices from the training generator
class_indices = valid_gen.classes  # Get class labels

# Count the occurrences of each class
class_counts = Counter(class_indices)

# Get class names and counts
class_names = list(class_counts.keys())
counts = list(class_counts.values())

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(class_names, counts, color='skyblue')
plt.title('Class Distribution in Training Data')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

test_gen=ImageDataGenerator(
    rescale=1./255
    ).flow_from_dataframe(
                                      dataframe=valid_csv,
                                      directory= valid_dir,
                                      x_col="filename",
                                      y_col="class",
                                      class_mode="categorical",
                                      shuffle=True,
                                      color_mode='rgb',
                                      batch_size=BATCH_SIZE,
                                      target_size=(WIDTH,HEIGHT)
                          )
classes_test=list(test_gen.class_indices.keys())
plt.figure(figsize=(10,6))
for X_batch, y_batch in test_gen:
    for i in range(10,8):
        plt.subplot(2,4,i+1)
        plt.imshow(X_batch[i])
        plt.title(classes_test[np.where(y_batch[55]==1)[0][0]])
        plt.grid(None)
    plt.show()
    break

from collections import Counter

# Assuming you have train_gen set up using ImageDataGenerator
# This will give you a list of class indices from the training generator
class_indices = test_gen.classes  # Get class labels

# Count the occurrences of each class
class_counts = Counter(class_indices)

# Get class names and counts
class_names = list(class_counts.keys())
counts = list(class_counts.values())

# Plotting
plt.figure(figsize=(20, 6))
plt.bar(class_names, counts, color='skyblue')
plt.title('Class Distribution in Training Data')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

model_path="/kaggle/working/models/"

from sklearn.metrics import precision_score, recall_score
def Evaluate_model_2(model, train_generator, valid_generator, test_generator):

        model_evaluate_train = model.evaluate(train_generator)
        print("Training Loss : ",model_evaluate_train[0])
        print("Training Accuracy : ",model_evaluate_train[1])
        print("Training Precision : ", model_evaluate_train[2])
        print("Training Recall :", model_evaluate_train[3])

        model_evaluate_valid = model.evaluate(valid_generator)
        print("Validation Loss : ",model_evaluate_valid[0])
        print("Validation Accuracy : ",model_evaluate_valid[1])
        print("Validation Precision : ", model_evaluate_valid[2])
        print("Validation Recall :",model_evaluate_valid[3])

        model_evaluate_test = model.evaluate(test_generator)
        print("Tesing Loss : ",model_evaluate_test[0])
        print("Tesing Accuracy : ",model_evaluate_test[1])
        print("Tesing Precision : ",model_evaluate_test[2])
        print("Tesing Recall :",model_evaluate_test[3])

        return np.round(model_evaluate_train[0],2),np.round(model_evaluate_test[0],2),\
               np.round(model_evaluate_train[1],2),np.round(model_evaluate_test[1],2),\
               np.round(model_evaluate_train[2],2),np.round(model_evaluate_test[2],2),\
               np.round(model_evaluate_train[3],2),np.round(model_evaluate_test[3],2)

"""# **VGG16 Model**"""

learn_rate=keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    factor=0.1,
    min_lr=1e-5
)
early_stop=keras.callbacks.EarlyStopping(
    patience=10,
    monitor="val_loss",
    restore_best_weights=True,
    verbose=1
)
checkpoint=keras.callbacks.ModelCheckpoint(
    filepath=model_path+"model_cp.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

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

"""# **VGG16 Evaluation**"""

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

"""# **CNN Model**"""

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

"""# **CNN Evaluation**"""

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

"""# **Predictions**"""

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