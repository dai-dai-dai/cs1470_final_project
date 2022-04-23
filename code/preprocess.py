import os
import enum
from scipy.misc import imread
import tensorflow as tf

class ArtGenre(enum.Enum):
    abstract_expressionism = 0
    baroque = 1
    cubism = 2
    fauvism = 3
    high_renaissance = 4
    iconoclasm = 5
    impressionism = 6
    old_greek_pottery = 7
    realism = 8
    rococo = 9
    romanticism = 10
    surrealism = 11

def one_hot(enum):
    tf.zeros()

def get_data(dir):
    # go through each folder in data/Pandora_V1
    # get all images and add labels
    # return: list of 2D tensors (images), list of one hot encoded vectors (labels)
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for subdirectory in dirs:
            subdir_path = os.listdir(os.path.join(root, subdirectory))
            for img in subdir_path:
                filename = os.path.join(subdir_path, img)
                images.append(tf.convert_to_tensor(imread(filename)))
                labels.append(one_hot())
    # ordered list of images, labels
    # shuffle
    # split to train and test 80/20


def main():
    print("start")
    preprocess("data")

if __name__ == '__main__':
	main()
