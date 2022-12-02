import os
import csv
import joblib
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

current_path = './facial/datasets/processed'
csv_path = current_path + '/dataset.csv'

target_names = ['angry', 'fear', 'happy', 'sad', 'surprise', 'neutral']
tick_marks = np.array(range(len(target_names))) + 0.5

def get_dataset(dataset_path):
    images = []
    labels = []
    pixels = []
    with open(dataset_path) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        for i, (label, pixel) in enumerate(csvr):
            img = np.asarray([float(p) for p in pixel.split()])
            images.append(img)
            labels.append(label)

    print(dataset_path+" --> Length of data: "+str(len(labels)))
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

from sklearn.decomposition import PCA

# Load data
images, labels = get_dataset(csv_path)

# Split original dataset to train & test (80%-20%)
train_img, test_img, train_label, test_label = train_test_split(images, labels,
  train_size=0.8, shuffle=True, random_state=42)

# Models chosen in this project
import warnings
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
models = []
models.append(('DUMMY', DummyClassifier(strategy='prior')))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier()))

from sklearn.model_selection import GridSearchCV

# Grid search cross-validation for choosing best parameters
param_grid_knn = {'n_neighbors':[3,4,5,6,7]}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn)
grid_knn.fit(train_img,train_label)
print("KNN: ",grid_knn.best_params_)

param_grid_svm = {'gamma':[0.0001,0.001,0.01], 'C':[0.01,1,10]}
grid_svm = GridSearchCV(SVC(), param_grid_svm, n_jobs=-1)
grid_svm.fit(train_img,train_label)
print("SVM: ",grid_svm.best_params_)

def classifer_model(model, model_name, save_model=False, plot_cr=False, plot_cm=False):
  # Train model
  print("Training...")
  model.fit(train_img, train_label)

  # Model result
  acc = model.score(test_img, test_label)
  print(' Acc: ', acc)
  print('=== Model training completed. ===')

  if save_model:
    out_name = './' + model_name + '.model'
    joblib.dump(model, out_name)
    print('=== Model has been saved. ===')

  # Plot classification report
  y_pred = model.predict(test_img)
  y_true = test_label

  if plot_cr:
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=target_names))

  if plot_cm:
    # Plot confusion matrix
    # %matplotlib inline
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

classifer_model(DummyClassifier(strategy='prior'), 'dummy', False, True, True)

classifer_model(SVC(), 'svc', True, True, True)

classifer_model(KNeighborsClassifier(n_neighbors=6), 'knn', True, True, True)
