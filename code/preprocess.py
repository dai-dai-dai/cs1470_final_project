import os
import tensorflow as tf
import numpy as np
import cv2

NUM_CLASSES = 12
TARGET_HEIGHT = 100
TARGET_WIDTH = 100
# smallest (117, 249)
# largest (3000, 2530)
# average (635, 673)

genre_to_index = {
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

def is_jpg(filepath):
    return filepath.endswith(".jpg")

def get_files(dir):
    filepaths = []
    for root, _, files in os.walk(dir, topdown = False):
        for file in files:
            if is_jpg(file):
                filepaths.append(os.path.join(root, file)) 
    return filepaths

def get_data(dir):
    """ 
        Get data from data folder, which is organised in folders by genre
        ::params::
            dir: path to data folder from root directory
        ::returns::
            train_images: list (num training images, (600,600,3))
            train_labels: list (num training images, (num classes))
            test_images: list (num testing images, (600,600,3))
            test_labels: list (num testing images, (num classes))
    """

    images = []
    labels = []
    
    # genre_folder_paths = [x[0] for x in os.walk(dir)] # list of str paths (data/realism... etc)
    # for genre_folder in genre_folder_paths[1:]: # ignore first (root)
    #     genre = genre_folder.split('/')[1]
    #     print("genre:", genre)
    #     index = genre_to_index[genre] # index associated with genre (0-11)
    #     for image_path in os.listdir(genre_folder): # loop paintings
    #         if image_path.endswith(".jpg"):
    #             filename = os.path.join(genre_folder, image_path)
    #             image = cv2.imread(filename)
    #             if image is not None:
    #                 resized = tf.image.resize_with_crop_or_pad(image / 255.0, TARGET_HEIGHT, TARGET_WIDTH)
    #                 images.append(tf.convert_to_tensor(resized))
    #                 labels.append(tf.one_hot(index, NUM_CLASSES))
    #     print("____________________________________")

    filepaths = get_files(dir)
    indices = np.arange(len(filepaths))
    np.random.shuffle(indices)
    for i in indices:
        filepath = filepaths[i]
        genre = filepath.split('/')[1]
        genre_index = genre_to_index[genre] # index associated with genre (0-11)
        image = cv2.imread(filepath)
        if image is not None:
            resized = tf.image.resize_with_crop_or_pad(image / 255.0, TARGET_HEIGHT, TARGET_WIDTH)
            images.append(tf.convert_to_tensor(resized))
            labels.append(tf.one_hot(genre_index, NUM_CLASSES))

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
    
    return train_images, train_labels, test_images, test_labels

def main():
    print("start")
    get_data("data")

if __name__ == '__main__':
	main()
