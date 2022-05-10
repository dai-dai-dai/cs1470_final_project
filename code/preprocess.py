import os
import tensorflow as tf
import numpy as np
import cv2

# ------------------------ EXPORTED VARIABLES ------------------------

# smallest (117, 249)
# largest (3000, 2530)
# average (635, 673)
NUM_SAMPLES = 5
NUM_CLASSES = 12
TARGET_HEIGHT = 150
TARGET_WIDTH = 150

GENRE_TO_INDEX = {
        "abstract_expressionism": 0,
        "baroque": 1,
        "cubism": 2,
        "fauvism": 3,
        "high_renaissance": 4,
        "iconoclasm": 5,
        "impressionism": 6,
        "old_greek_pottery": 7,
        "realism": 8,
        "rococo": 9,
        "romanticism": 10,
        "surrealism": 11
    }

INDEX_TO_GENRE = {v: k for k, v in GENRE_TO_INDEX.items()}


# ------------------------ UTIL FUNCTIONS ------------------------

def is_jpg(file):
    """
        Determines whether filepath is .jpg
        ::params::
            str file: file path name
        ::returns::
            True if filepath is .jpg, false otherwise
    """
    return file.endswith(".jpg")

def get_files(dir):
    """
        Gets all files in directory
        ::params::
            str dir: directory path name
        ::returns::
            filepaths: list of all filepath names in directory
    """
    filepaths = []
    for root, _, files in os.walk(dir, topdown = False):
        for file in files:
            if is_jpg(file):
                filepaths.append(os.path.join(root, file)) 
    return filepaths


# ------------------------ PREPROCESSING FUNCTION ------------------------

def get_data(dir):
    """ 
        Get data from data folder, which is organised in folders by genre
        - training and testing data: used to train and test the model
        - sampling data: a random sample to get a sense of how the model predicts specific images
        - data by category: the data organized by category to calculate categorical accuracy and prediction distribution per genre
        ::params::
            dir: path to data folder from root directory
        ::returns::
            train_images: list (num training images, (TARGET_WIDTH,TARGET_HEGITH,3))
            train_labels: list (num training images, (NUM_CLASSES))
            test_images: list (num testing images, (TARGET_WIDTH,TARGET_HEGITH,3))
            test_labels: list (num testing images, (NUM_CLASSES))
            sample_files: at each index, contains a list of five random image paths of the genre
            sample_images: at each index, contains a list of corresponding images
            sample_labels: at each index, contains a list of corresponding labels
            images_by_category: at each index, contains a list of all images belonging to the genre
            matching_labels: at each index, contains a list of corresponding labels
    """

    # training and testing data
    images = []
    labels = []
    # sampling data
    sample_files = []
    sample_images = []
    sample_labels = []
    # data by category
    images_by_category = []
    matching_labels = []
    for i in range(NUM_CLASSES):
        sample_files.append([])
        sample_images.append([])
        sample_labels.append([])
        images_by_category.append([])
        matching_labels.append([])

    filepaths = get_files(dir)
    indices = np.arange(len(filepaths))
    np.random.shuffle(indices)
    for i in indices:
        filepath = filepaths[i]
        genre = filepath.split('/')[1]
        genre_index = GENRE_TO_INDEX[genre] # index associated with genre (0-11)
        image = cv2.imread(filepath)
        if image is not None:
            # construct image & label tensors
            resized = tf.convert_to_tensor(tf.image.resize_with_crop_or_pad(image / 255.0, TARGET_HEIGHT, TARGET_WIDTH))
            label = tf.one_hot(genre_index, NUM_CLASSES)
            # training and testing data
            images.append(resized)
            labels.append(label)
            # data by category
            images_by_category[genre_index].append(resized)
            matching_labels[genre_index].append(label)
            # sampling data
            if len(sample_images[genre_index]) < NUM_SAMPLES:
                sample_files[genre_index].append(filepath)
                sample_images[genre_index].append(resized)
                sample_labels[genre_index].append(label)

    # test display
    # cv2.imshow('sample image',np.asarray(images[5]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    # shuffle
    # indices = np.arange(len(images))
    # np.random.shuffle(indices)
    # images = tf.gather(images, indices)
    # labels = tf.gather(labels, indices)
    # print('image sample: ', images[1])
    # print('labels sample: ', labels[0:10])
    
    # split to train and test 80/20
    train_len = len(images) * 4 // 5
    train_images, train_labels = images[:train_len], labels[:train_len]
    test_images, test_labels = images[train_len:], labels[train_len:]
    
    return train_images, train_labels, test_images, test_labels, sample_files, sample_images, sample_labels, images_by_category, matching_labels
