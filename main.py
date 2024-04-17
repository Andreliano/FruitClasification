import numpy as np
from sklearn import neural_network

from utils import describeDataset, loadImageData, split_data, plot_histogram, normalisation, \
    convertPixelsMatrixIntoGrayScaleArray, training, classification, evalMultiClass, createModel, plotConfusionMatrix, \
    createGoogleNetModel

if __name__ == "__main__":
    inputs, outputs = loadImageData('dataset/Augmented Image')

    # inputs_transformed = convertPixelsMatrixIntoGrayScaleArray(inputs)
    # describeDataset(inputs_transformed, outputs)
    # plot_histogram(train_outputs, 'Histogram of labels for train inputs')
    # plot_histogram(test_outputs, 'Histogram of labels for test inputs')

    # train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs_transformed, outputs)
    # train_inputs_normalized, test_inputs_normalized = normalisation(train_inputs, test_inputs, 224 * 224 * 3)
    # classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=200,
    #                                           solver='sgd',
    #                                           verbose=10, random_state=1, learning_rate_init=.1)
    # classifier = training(classifier, train_inputs_normalized, train_outputs)
    # computed_test_outputs = classification(classifier, test_inputs_normalized)
    # print(computed_test_outputs)
    # print(test_outputs)
    # accuracy, precision, recall, cm = evalMultiClass(np.array(test_outputs), computed_test_outputs,
    #                                                  ['FreshApple', 'FreshBanana', 'FreshGrape', 'FreshGuava',
    #                                                   'FreshJujube', 'FreshOrange',
    #                                                   'FreshPomegranate', 'FreshStrawberry', 'RottenApple',
    #                                                   'RottenBanana', 'RottenGrape', 'RottenGuava',
    #                                                   'RottenJujube', 'RottenOrange', 'RottenPomegranate',
    #                                                   'RottenStrawberry'])
    # print("ANN image:")
    # print('acc: ', accuracy)
    # print('precision: ', precision)
    # print('recall: ', recall)

    output_labels = set(outputs)
    train_inputs, train_outputs, test_inputs, test_outputs = split_data(inputs, outputs)
    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)
    model = createGoogleNetModel()
    output_labels = sorted(output_labels)
    output_to_index = {label: i for i, label in enumerate(output_labels)}
    print(output_to_index)
    index_train_outputs = [output_to_index[output] for output in train_outputs]
    print(train_inputs, np.array(index_train_outputs))
    model.fit(train_inputs, np.array(index_train_outputs), epochs=15, batch_size=32)

    test_inputs = np.array(test_inputs)
    computed_test_outputs = model.predict(test_inputs)
    computed_test_outputs = np.argmax(computed_test_outputs, axis=1)

    output_to_index = {label: i for i, label in enumerate(set(test_outputs))}
    index_test_outputs = [output_to_index[output] for output in test_outputs]
    accuracy, precision, recall, cm = evalMultiClass(np.array(index_test_outputs), computed_test_outputs,
                                                     [0, 1, 2, 3,
                                                      4, 5,
                                                      6, 7, 8,
                                                      9, 10, 11,
                                                      12, 13, 14,
                                                      15])

    print("CNN image:")
    print('acc: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    plotConfusionMatrix(cm, [0, 1, 2, 3,
                             4, 5,
                             6, 7, 8,
                             9, 10, 11,
                             12, 13, 14,
                             15], "fruit classification", output_labels)
