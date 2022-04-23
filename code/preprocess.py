import os
import enum
import PIL
import tensorflow as tf
import numpy as np
import cv2

NUM_CLASSES = 12
TARGET_HEIGHT = 600
TARGET_WIDTH = 600
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

def get_data(dir):
    """ 
        Get data from data folder, which is organised in folders by genre
        :params:
            dir: path to data folder from root directory
        :returns:
            train_images: list (num training images, (600,600,3))
            train_labels: list (num training images, (num classes))
            test_images: list (num testing images, (600,600,3))
            test_labels: list (num testing images, (num classes))
    """

    images = []
    labels = []
    skip = True
    
    for root, subdirs, files in os.walk(dir):
        if skip:
            skip = False
            continue
        print('root:', root)
        genre_dir = os.listdir(root)
        index = genre_to_index[root.split('/')[1]]
        for img in genre_dir:
            if img.endswith(".jpg"):
                filename = os.path.join(os.path.join(root, img))
                image = cv2.imread(filename)
                if image is not None:
                    resized = tf.image.resize_with_pad(image, TARGET_HEIGHT, TARGET_WIDTH, antialias=True)
                    images.append(tf.convert_to_tensor(resized))
                    labels.append(tf.one_hot(index, NUM_CLASSES))
        print("____________________________________")

    # shuffle
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = tf.gather(images, indices)
    labels = tf.gather(labels, indices)
    print('image sample: ', images[0])
    print('labels sample: ', labels[0:10])
    
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
