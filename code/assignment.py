import tensorflow as tf
import numpy as np
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

        input_shape = (self.batch_size, None, None, 3) # to account for different shaped images
        self.layers = tf.keras.Sequential(
            tf.keras.layers.conv2D(
                filters=8, 
                kernel_size=4,
                activation='relu',
                input_shape=input_shape[1:],
                padding='same'),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(.25),
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(.25),
            tf.keras.layers.Dense(self.hidden_dim, activation='softmax')
        )

    def call(self, inputs):
        logits = self.layers(inputs)
        return logits

    def loss(self, logits, labels):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        avg_loss = tf.math.reduce_mean(loss)
        return avg_loss
    
    def accuracy(self, logits, labels):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#-------------------------------------------

def train(model, train_inputs, train_labels):
        ind = tf.random.shuffle(np.array(range(0, len(train_inputs))))
        shuffled_inputs = tf.gather(train_inputs, ind)
        shuffled_labels = tf.gather(train_labels, ind)
        
        for i in range(0, int(len(train_inputs)/model.batch_size)):
            inputs = shuffled_inputs[i*model.batch_size: (i + 1)*model.batch_size]
            #inputs = tf.image.random_flip_left_right(inputs)
            
            labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

            with tf.GradientTape() as gt:
                logits = model.call(inputs)
                loss = model.loss(logits, labels)
                model.loss_list.append(loss)
            
            grad = gt.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        
        return

def test(model, test_inputs, test_labels):
        logits = model.call(test_inputs, is_testing=True)
        test_accuracy = model.accuracy(logits, test_labels)

        return test_accuracy

def main():
    train_inputs, train_label, test_inputs, test_labels = get_data('data')

    model=Art_Model()

    for i in range(1, model.epochs):
        print(f'epoch: {1}')
        train(model, train_inputs, train_label)

    accuracy = test(model, test_inputs, test_labels)
    print(f'model accuracy: {accuracy}')
    pass






