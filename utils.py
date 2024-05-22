import itertools

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from matplotlib import pyplot as plt


def describeDataset(inputs, outputs):
    data = {'Pixel intensity': []}

    for input, output in zip(inputs, outputs):
        pixel_intensity_mean = np.mean(input)
        data['Pixel intensity'].append(pixel_intensity_mean)

    # for subdirectory in os.listdir('dataset/Original Image'):
    #     for img_name in os.listdir(f'dataset/Original Image/{subdirectory}')[:1]:
    #         print(img_name)
    #         img_path = os.path.join(f'dataset/Original Image/{subdirectory}', img_name)
    #         img = Image.open(img_path)
    #         img_array = np.array(img.getdata())
    #
    #         width, height = img.size
    #         pixel_intensity_mean = np.mean(img_array)
    #
    #         data['Width'].append(width)
    #         data['Height'].append(height)
    #         data['Pixel intensity'].append(pixel_intensity_mean)

    df = pd.DataFrame(data)
    description = df.describe()
    print(description)


# def loadImageData(path):
#     inputs = []
#     outputs = []
#     for subdirectory in os.listdir(path):
#         for img_name in os.listdir(f'{path}/{subdirectory}')[:5]:
#             if img_name.endswith("jpg"):
#                 print(subdirectory, img_name)
#                 imgNormal = Image.open(f'{path}/{subdirectory}/{img_name}')
#                 imgNormal = imgNormal.resize((224, 224))
#                 imgNormal = img_to_array(imgNormal)
#                 imgNormal = preprocess_input(imgNormal)
#                 inputs.append(imgNormal)
#                 outputs.append(subdirectory)
#
#     return inputs, outputs

def loadImageData(path):
    inputs = []
    outputs = []
    for subdirectory in os.listdir(path):
        for img_name in os.listdir(f'{path}/{subdirectory}')[:10]:
            if img_name.endswith("jpg"):
                print(subdirectory, img_name)
                imgNormal = Image.open(f'{path}/{subdirectory}/{img_name}')
                imgNormal = imgNormal.resize((224, 224))
                pixelMatrixNormal = np.array(imgNormal.getdata())
                inputs.append(pixelMatrixNormal)
                outputs.append(subdirectory)

    return inputs, outputs


def split_data(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    train_indexes = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    test_indexes = [i for i in indexes if not i in train_indexes]

    train_inputs = [inputs[i] for i in train_indexes]
    train_outputs = [outputs[i] for i in train_indexes]
    test_inputs = [inputs[i] for i in test_indexes]
    test_outputs = [outputs[i] for i in test_indexes]

    return train_inputs, train_outputs, test_inputs, test_outputs


def convertPixelsMatrixIntoGrayScaleArray(inputs):
    gray_arrays = []
    for pixels in inputs:
        scaled_pixels = (pixels * 255.0 / pixels.max()).astype(np.uint8)
        img = Image.fromarray(scaled_pixels, mode='L')
        gray_array = np.reshape(np.asarray(img.getdata()), -1)
        gray_arrays.append(gray_array)
    return np.asarray(gray_arrays)


def get_mean_and_standard_deviation(data):
    mean = sum(data) / len(data)
    sum_differences = 0
    for x in data:
        sum_differences = sum_differences + (x - mean) ** 2
    standard_deviation = (sum_differences / (len(data) - 1)) ** 0.5
    return mean, standard_deviation


def normalisation(train_inputs, test_inputs, number_of_features):
    number_of_train_inputs = len(train_inputs)
    parameters_for_normalization = []
    for j in range(number_of_features):
        feature = [train_inputs[i][j] for i in range(number_of_train_inputs)]
        mean, standard_deviation = get_mean_and_standard_deviation(feature)
        parameters_for_normalization.append([mean, standard_deviation])
    train_inputs_normalized = []
    for inputs in train_inputs:
        features_normalized = []
        for i in range(len(inputs)):
            if parameters_for_normalization[i][1] == 0:
                normalized_value = 0
            else:
                normalized_value = (inputs[i] - parameters_for_normalization[i][0]) / parameters_for_normalization[i][1]
            features_normalized.append(normalized_value)
        train_inputs_normalized.append(features_normalized)

    test_inputs_normalized = []
    for inputs in test_inputs:
        features_normalized = []
        for i in range(len(inputs)):
            if parameters_for_normalization[i][1] == 0:
                normalized_value = 0
            else:
                normalized_value = (inputs[i] - parameters_for_normalization[i][0]) / parameters_for_normalization[i][1]
            features_normalized.append(normalized_value)
        test_inputs_normalized.append(features_normalized)
    return train_inputs_normalized, test_inputs_normalized


def createModel():
    model = tf.keras.Sequential([
        keras.layers.Reshape((224, 224, 3), input_shape=(50176, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(16, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def createGoogleNetModel():
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(16, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def createAlexNetModel(input_shape, num_classes):
    model = Sequential()


    model.add(Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))


    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))


    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))


    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def training(classifier, trainInputsNormalized, trainOutputs):
    classifier.fit(trainInputsNormalized, trainOutputs)
    return classifier


def classification(classifier, validationInputsNormalized):
    computedValidationOutputs = classifier.predict(validationInputsNormalized)
    return computedValidationOutputs


def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels, labels=labelNames)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / (sum([confMatrix[j][i] for j in range(len(labelNames))]) + 1e-9)
        recall[labelNames[i]] = confMatrix[i][i] / (sum([confMatrix[i][j] for j in range(len(labelNames))]) + 1e-9)
    return acc, precision, recall, confMatrix


def plot_histogram(outputs, histogram_name):
    outputs_labels = set(outputs)
    output_to_index = {label: i for i, label in enumerate(outputs_labels)}
    index_outputs = [output_to_index[output] for output in outputs]

    plt.hist(index_outputs, bins=range(len(outputs_labels) + 1), align='left', rwidth=0.8)
    legend_patches = [mpatches.Patch(label=f'{label} = {output_to_index[label]}') for label in outputs_labels]
    plt.legend(handles=legend_patches)
    plt.title(histogram_name)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(range(len(outputs_labels)), labels=[f'{i}' for label, i in output_to_index.items()])
    plt.show()


def plotConfusionMatrix(cm, class_names, title, class_descriptions=None):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if class_descriptions:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{class_names[i]}: {desc}',
                                  markerfacecolor='g', markersize=5) for i, desc in enumerate(class_descriptions)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1), loc='upper left')

    plt.show()
