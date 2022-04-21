import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

class Art_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Art_Model, self).__init__()
        self.batch_size = 100
        self.num_classes = num_classes

        self.hidden_dim = 100
        self.learning_rate = .01
        self.epochs = None

    def call(self, inputs):
        pass

    def train(model, train_inputs, train_labels):
        pass

    def test(model, test_inputs, test_labels):
        pass






