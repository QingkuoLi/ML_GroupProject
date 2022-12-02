# %load_ext tensorboard
import os
import csv
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import SGD
# from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix

# Some global parameters
batch_size  = 64
num_classes = 6
nb_epoch    = 60
img_height  = 48
img_width   = 48
img_channel = 1

current_path = '.'
csv_path = current_path + '/dataset.csv'

target_names = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
tick_marks = np.array(range(len(target_names))) + 0.5

def get_dataset(dataset_path):
    images = []
    labels = []
    with open(dataset_path) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
            img = Image.fromarray(pixel).convert('L')
            img = img_to_array(img)
            images.append(img)
            labels.append(label)

    print(dataset_path+" --> Length of data: "+str(len(labels)))
    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=num_classes)

    return images, labels

class CNNModel:
    def __init__(self):
        # Model architecture
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.summary() # Print layer arch
    
    def train_model(self, plot_cr=False, plot_cm=False):
        # Load data
        # Split original dataset to train, val & test (80%-10%-10%)
        images, labels = get_dataset(csv_path)
        train_img, rest_img, train_label, rest_label = train_test_split(images, labels,
            train_size=0.8, shuffle=True, random_state=42)
        val_img, test_img, val_label, test_label = train_test_split(rest_img, rest_label,
            test_size=0.5, shuffle=True, random_state=42)
        # train_img, test_img, train_label, test_label = train_test_split(images, labels,
        #     train_size=0.80, shuffle=True, random_state=42)

        # Stengthen data
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=5,
                                           zoom_range=0.05,
                                           brightness_range=[0.9,1.1],
                                           horizontal_flip=True)
        val_datagen  = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow(
            train_img, train_label,
            batch_size=batch_size,
            shuffle=True)
        val_generator   = val_datagen.flow(
            val_img, val_label,
            batch_size=batch_size,
            shuffle=False)
        test_generator  = test_datagen.flow(
            test_img, test_label,
            batch_size=batch_size,
            shuffle=False)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                    #   optimizer=Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                      metrics=['accuracy'])
        
        log_path = current_path + '/logs'
        logdir = os.path.join(log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        # Train model
        print("Training...")
        self.model.fit(train_generator,
                       steps_per_epoch=len(train_label)/(batch_size),
                       epochs=nb_epoch,
                       validation_data=val_generator,
                       validation_steps=len(val_label)/(batch_size),
                       callbacks=[tensorboard_callback])
        # Model result
        self.model_evaluate = self.model.evaluate(test_generator, steps=len(test_label)/batch_size)
        
        print('Loss: ', self.model_evaluate[0])
        print(' Acc: ', self.model_evaluate[1])
        print('=== Model training completed. ===')

        if plot_cr: # Plot classification report
          Y_pred = self.model.predict(test_generator)
          y_pred = np.argmax(Y_pred, axis=-1)
          y_true = [np.argmax(one_hot)for one_hot in test_generator.y]

          print('Classification Report:')
          print(classification_report(y_true, y_pred, target_names=target_names))
        
        if plot_cm: # Plot confusion matrix
          Y_pred = self.model.predict(test_generator)
          y_pred = np.argmax(Y_pred, axis=-1)
          y_true = [np.argmax(one_hot)for one_hot in test_generator.y]
        #   %matplotlib inline
          cm = confusion_matrix(y_true, y_pred)
          print(cm)
          np.set_printoptions(precision=3)
          cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          print(cm_normalized)
          plt.figure(figsize=(12, 8), dpi=120)

          ind_array = np.arange(len(target_names))
          x, y = np.meshgrid(ind_array, ind_array)

          for x_val, y_val in zip(x.flatten(), y.flatten()):
              c = cm_normalized[y_val][x_val]
              plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
          # offset the tick
          plt.gca().set_xticks(tick_marks, minor=True)
          plt.gca().set_yticks(tick_marks, minor=True)
          plt.gca().xaxis.set_ticks_position('none')
          plt.gca().yaxis.set_ticks_position('none')
          plt.grid(True, which='minor', linestyle='-')
          plt.gcf().subplots_adjust(bottom=0.15)

          plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
          plt.title('Normalized confusion matrix')
          plt.colorbar()
          xlocations = np.array(range(len(target_names)))
          plt.xticks(xlocations, target_names)
          plt.yticks(xlocations, target_names)
          plt.ylabel('True label')
          plt.xlabel('Predicted label')
          plt.show()

    def save_model(self):
        # save model file
        model_json = self.model.to_json()
        with open(current_path+"/model_json_.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(current_path+'/model_weight_.h5')
        self.model.save(current_path+'/facial_model.h5')
        print('=== Model has been saved. ===')

if __name__ == '__main__':
    facialExpModel = CNNModel()
    facialExpModel.train_model()
    facialExpModel.save_model()