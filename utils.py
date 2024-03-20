import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
import os

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


def loadImageData(path):
    inputs = []
    outputs = []
    for subdirectory in os.listdir(path):
        for img_name in os.listdir(f'{path}/{subdirectory}')[:20]:
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