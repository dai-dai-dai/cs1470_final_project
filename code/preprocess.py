import os
import enum
import PIL
import tensorflow as tf

NUM_CLASSES = 12
TARGET_HEIGHT = 600
TARGET_WIDTH = 600

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

# smallest (117, 249)
# largest (3000, 2530)
# average (635, 673)

def get_data(dir):
    # go through each folder in data
    # get all images and add labels
    # return: list of 2D tensors (images), list of one hot encoded vectors (labels)

    images = []
    labels = []
    print("hi")
    skip = True
    for root, subdirs, files in os.walk(dir):
        if skip:
            skip = False
            continue
        # # shape: (# images, # classes)
        genre_dir = os.listdir(root)
        print("root:",root)
        index = genre_to_index[root.split('/')[1]]
        print('index:', index)
        for img in genre_dir:
            if img.endswith(".jpg"):
                filename = os.path.join(os.path.join(root, img))
                try:
                    image = PIL.Image.open(filename)
                except PIL.UnidentifiedImageError:
                    print(filename)
                    continue
                resized = tf.image.resize_with_pad(image, TARGET_HEIGHT, TARGET_WIDTH, antialias=True)
                images.append(tf.convert_to_tensor(resized))
                label = tf.zeros([NUM_CLASSES,])
                label[index] = 1
                labels.append(label)

    
    # ordered list of images, labels
    # shuffle
    # split to train and test 80/20


def main():
    print("start")
    get_data("data")

if __name__ == '__main__':
	main()
