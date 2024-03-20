from utils import describeDataset, loadImageData, split_data, plot_histogram, normalisation, \
    convertPixelsMatrixIntoGrayScaleArray

if __name__ == "__main__":
    inputs, outputs = loadImageData('dataset/Augmented Image')
    inputs_transformed = convertPixelsMatrixIntoGrayScaleArray(inputs)
    describeDataset(inputs_transformed, outputs)
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs_transformed, outputs)
    print(train_inputs[0])
    plot_histogram(train_outputs, 'Histogram of labels for train inputs')
    plot_histogram(test_outputs, 'Histogram of labels for test inputs')
    train_inputs_normalized, test_inputs_normalized = normalisation(train_inputs, test_inputs, 150528)
    print(train_inputs_normalized[0])