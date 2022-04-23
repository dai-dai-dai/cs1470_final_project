import os
import enum
import PIL
import tensorflow as tf

NUM_CLASSES = 12
TARGET_HEIGHT = 600
TARGET_WIDTH = 600

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

# smallest (117, 249)
# largest (3000, 2530)
# average (635, 673)

def get_data(dir):
    # go through each folder in data
    # get all images and add labels
    # return: list of 2D tensors (images), list of one hot encoded vectors (labels)

    images = []
    len_genre_dir = []
    for root, subdirs, files in os.walk(dir):
        # # shape: (# images, # classes)
        genre_dir = os.listdir(root)
        len_genre_dir.append(len(genre_dir))
        for img in genre_dir:
            filename = os.path.join(os.path.join(root, img))
            try:
                image = PIL.Image.open(filename)
            except PIL.UnidentifiedImageError:
                len_genre_dir[-1] -= 1
                print(filename)
                continue
            ## resize here with pad for consistency
            # resize w/ pad
            resized = tf.image.resize_with_pad(image, )
            images.append(tf.convert_to_tensor(image))

    # ordered list of images, labels
    # shuffle
    # split to train and test 80/20


def main():
    print("start")
    get_data("data")

if __name__ == '__main__':
	main()
