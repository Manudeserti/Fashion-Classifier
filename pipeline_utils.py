import os
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
def read_img_file_name(path, class_labels):
    images = []
    for class_label in class_labels:
        images.append(glob.glob(os.path.join(path, class_label, '*.jpg')))
    return [val for sublist in images for val in sublist]

def build_model(vgg_conv, num_topics, drop_rate, learning_rate):
    model = keras.models.Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())

    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(BatchNormalization())

    model.add(Dense(2524, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(num_topics, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

def data_aug(train_path, validation_path, batch_size, batch_size_val, shift_fraction, img_r, img_c):
    gen = ImageDataGenerator(
          rescale=1./255,
          height_shift_range=shift_fraction,
          width_shift_range=shift_fraction,
          shear_range=shift_fraction,
          zoom_range=shift_fraction,
          horizontal_flip=True,
          vertical_flip=True)

    val_gen = ImageDataGenerator(rescale=1./255)

    batches = gen.flow_from_directory(train_path,
          target_size=(img_r, img_c),
          batch_size=batch_size,
          class_mode='categorical',
          shuffle=True)
          
    val_batches = val_gen.flow_from_directory(validation_path,
          target_size=(img_r, img_c),
          batch_size=batch_size_val,
          class_mode='categorical',
          shuffle=False)
          
    return batches, val_batches

def fit_model(model, batches, val_batches, batch_size, val_batch_size, epochs):
    steps_per_epoch = len(batches.filenames) // batch_size
    validation_steps = len(val_batches.filenames) // val_batch_size

    history = model.fit(
        batches,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_batches,
        validation_steps=validation_steps
    )
    score = model.evaluate(val_batches, verbose=0)
    return model, history, score

def curves(model_histories, epochs, output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)
    acc = model_histories.history.get('accuracy', model_histories.history.get('acc'))
    val_acc = model_histories.history.get('val_accuracy', model_histories.history.get('val_acc'))
    loss = model_histories.history.get('loss', model_histories.history.get('train_loss'))
    val_loss = model_histories.history.get('val_loss')

    plt.figure()
    plt.plot(range(epochs), acc, 'm--', label='Training accuracy')
    plt.plot(range(epochs), val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'{output_dir}/CurvesAcc.jpg')

    plt.figure()
    plt.plot(range(epochs), loss, 'm--', label='Training loss')
    plt.plot(range(epochs), val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'{output_dir}/CurvesLoss.jpg')
    plt.close('all')

def evaluate_confusion_matrix(model, val_batches, class_labels, output_dir='./outputs'):
    os.makedirs(output_dir, exist_ok=True)
    Y_pred = model.predict(val_batches)
    y_pred_classes = np.argmax(Y_pred, axis=1)
    
    confusion_mat = confusion_matrix(val_batches.labels, y_pred_classes)
    classification_rep = classification_report(val_batches.labels, y_pred_classes, target_names=class_labels)
    
    plt.figure(figsize=(6,4))
    df_confusion_mat = pd.DataFrame(confusion_mat)
    sns.heatmap(df_confusion_mat, annot_kws={"size": 10}, linewidths=.5, cmap='PuBu', annot=True,
                yticklabels=class_labels, xticklabels=class_labels, fmt='g')
    plt.xticks(rotation=40)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.yticks(rotation=0)
    plt.savefig(f'{output_dir}/confusion_matrix.jpg')
    plt.close()
    
    return classification_rep