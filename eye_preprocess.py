import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves import cPickle as pickle
import cv2

openDirs = ['dataset/open_eyes']
closeDirs = ['dataset/closed_eyes']


def generate_dataset(type, dirData):
    dataset = np.ndarray([1231 * 2, 24, 24, 1], dtype='float32') #All Images into 24*24 matrix and push into object (All images into 1 pickle file) 
    i = 0
    for dir in dirData:
        for filename in os.listdir(dir):
            if filename.endswith('.jpg'):
                im = cv2.imread(dir + '/' + filename)
                # Convert to grayscale image
                im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255 #color images to grey scale
                dataset[i, :, :, :] = im[:, :, :]
                i += 1

    if type == 1:
        labels = np.ones([len(dataset), 1], dtype=int)
    else:
        labels = np.zeros([len(dataset), 1], dtype=int)
    return dataset, labels


def save_train_and_test_set(dataset, labels, ratio, pickle_file): # Split 80% training and 20% testing the datasent (1st split is test and remaining is train)
    split = int(len(dataset) * ratio) #Here ratio of 20:80
    train_dataset = dataset[:split]
    train_labels = labels[:split]
    test_dataset = dataset[split:]
    test_labels = labels[split:]

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL) #Keeps pushing the images in object and then save at the end
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)


# Main
if __name__ == '__main__':
    dataset_open, labels_open = generate_dataset(1, openDirs)
    dataset_closed, labels_closed = generate_dataset(0, closeDirs)

    ratio = 0.8 #Ratio of Training and Testing

    pickle_file_open = 'trained_model/open_eyes.pickle' #File saved in Trained Model Folder
    pickle_file_closed = 'trained_model/closed_eyes.pickle' #File saved in Trained Model Folder

    # Save open dataset to pickle file
    save_train_and_test_set(dataset_open, labels_open, ratio, pickle_file_open)
    # Save close dataset to pickle file
    save_train_and_test_set(dataset_closed, labels_closed, ratio, pickle_file_closed)
