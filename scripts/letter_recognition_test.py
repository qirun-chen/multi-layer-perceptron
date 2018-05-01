"""
A test for Multi Layer Perceptron

An assignment of Connectionist Computing - UCD COMP30230 Module

- Exceptional Test for letter recognition
"""

__version__ = '0.1'
__author__ = 'Qirun Chen - Student No. 16212138'
__date__ = '27 Apr 2018'

import csv
import numpy as np
from multi_layer_perceptron import MLP


def to_character(n):
    """ Map an integer to the character according to ASCII table
    :param n: The int number in ASCII
    :return: The character in ASCII
    """
    # The starting number of character 'A' in ASCII table
    char_delta = ord('A')
    return chr(int(n) + char_delta)


dataset = list()
# Read data from csv
with open('../data/letter-recognition.csv', newline='') as data_file:
    reader = csv.reader(data_file, delimiter=',')
    for row in reader:
        # Map the character to the integer number
        row[0] = ord(row[0]) - ord('A')
        dataset.append(row)

# Define a ratio to split the training test and test set
training_split = int(len(dataset) * 0.8)
# Split the training set
training_set = np.array(dataset[:training_split], dtype=np.int)
X = training_set[:, 1:]
y = training_set[:, 0]

# Define the number of output labels
num_labels = 26

# One hot encode y labels
categorical_y = np.zeros((len(y), num_labels))
for i, l in enumerate(y):
    categorical_y[i][l] = 1

# Normalize the features - Important!!!
X = X/15
# Initialize the MLP network
mlp = MLP(16, 30, 26, learning_rate=0.1, max_epochs=1300, verbose=1)
mlp.fit(X, categorical_y)

# Split the inputs and outputs of the test set
test_set = np.array(dataset[training_split:], dtype=np.int)
test_x = test_set[:, 1:]
# Normalization - Important!!!
test_x = test_x / 15
test_y = test_set[:, 0]
# Predict on the normalized test set
prediction = mlp.predict(test_x)

# Print prediction details
# Initialize the confusion dictionary
confusion_dict = {to_character(i): 0 for i in range(26)}
letter_num_dict = {to_character(i): 0 for i in range(26)}
for i, _ in enumerate(test_y):
    letter_num_dict[to_character(test_y[i])] += 1
    # Print some predictions
    if i % 300 == 0:
        print('Expected: %s | Output: %s' % (to_character(test_y[i]), to_character(prediction[i])))
    if test_y[i] == prediction[i]:
        confusion_dict[to_character(prediction[i])] += 1

print('==' * 20)
# Calculate the accuracy
accuracy = sum(confusion_dict.values()) / len(test_y)
print('Test sample size: %d | Correctly predicted sample size: %d' %
      (len(test_y), sum(confusion_dict.values())))
print('Accuracy: %.3f' % accuracy)

# Performance on each class
print('==' * 20)
for k, v in letter_num_dict.items():
    print('%s => Sample Number: %d | Correct Number: %d | Accuracy: %.3f' %
          (k, v, confusion_dict[k], confusion_dict[k] / v))


