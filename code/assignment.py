import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import \
    Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import Dense, Flatten, Reshape
from preprocess import get_data

class Art_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(Art_Model, self).__init__()

        #hyperparameters:
        self.batch_size = 100
        self.num_classes = num_classes

        self.hidden_dim = 100
        self.learning_rate = .01
        self.epochs = 10

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        input_shape = (self.batch_size, None, None, 3) #to account for different shaped images
        self.layers = tf.keras.Sequential(
            Conv2D(
                filters=8, 
                kernel_size=4,
                activation='relu',
                input_shape=input_shape[1:],
                padding='same'),
            Dense(self.hidden_dim, activation='relu'),
            Dropout(.25),
            Dense(self.hidden_dim, activation='relu'),
            Dropout(.25),
            Dense(12, activation='softmax')
            # things maybe to tweak:
            # - add maxpool
            # - tweak width & depth of convo layers
        )

    def call(self, inputs):
        """Performs forward pass through Art_Model by applying convolution
        and dense layers to inputs.
        
        Inputs:
        -inputs: Batch of input images of shape (N, H, W, 3)
        
        Returns:
        -logits: probability distribution over all the labels for each image
        of shape (N, C)"""
        logits = self.layers(inputs)
        return logits

    def loss(self, logits, labels):
        """ 
        Computes the cross entropy loss of the Art_Model. Returned loss is the average loss 
        per sample in the current batch

        Inputs:
        - logits: Probability distribution over classes for each image as calculated by the 
        call function. Shape: (N, H, W, 3)
        - labels: the true labels for classification of each image

        Returns:
        - loss: Tensor containing the containing the average loss for the batch
        """
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        avg_loss = tf.math.reduce_mean(loss)
        return avg_loss
    
    def accuracy(self, logits, labels):
        """
        Computes the accuracy of logit predictions

        Inputs: 
        - logits: Probability distribution over classes for each image as calculated by the 
        call function. Shape: (N, H, W, 3)
        - labels: the true labels for classification of each image

        Returns:
        - accuracy: a tensor containing the accuracy of the logit predictions as compared to 
        the true labels
        """
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#-------------------------------------------

def train(model, train_inputs, train_labels):
    """
    Trains the Art_Model for self.epochs epochs

    Inputs:
    - model: instance of Art_Model
    - train_inputs: Input images of shape (N, H, W, 3)
    - train_labels: one-hot encoded labels for each image of shape(N, C)

    Returns:
    - total_loss: Sum of loss values of all batches
    """
    ind = tf.random.shuffle(np.array(range(0, len(train_inputs))))
    shuffled_inputs = tf.gather(train_inputs, ind)
    shuffled_labels = tf.gather(train_labels, ind)

    total_loss = 0.0
    
    for i in range(0, int(len(train_inputs)/model.batch_size)):      
        inputs = shuffled_inputs[i*model.batch_size: (i + 1)*model.batch_size]
        #inputs = tf.image.random_flip_left_right(inputs)
        
        labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

        with tf.GradientTape() as gt:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)
            model.loss_list.append(loss)

            total_loss += loss
        
        grad = gt.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
    
    return total_loss

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels

    Inputs: 
    - model: instance of Art_Model
    - test_inputs: test images of shape (N, H, W, 3)
    - test_labels: test labels of shape (N, C)

    Returns:
    - test_accuracy: The average accuracy across all batches
    """
    logits = model.call(test_inputs, is_testing=True)
    test_accuracy = model.accuracy(logits, test_labels)

    return test_accuracy

def main():
    """ 
    Reads in image data (12 classes), initializes Art_Model, and trains model for 
    model.epochs epochs.

    Prints:
    - loss after each epoch
    - model accuracy at end of training 
    """
    train_inputs, train_label, test_inputs, test_labels = get_data('data')

    model=Art_Model()

    for i in range(1, model.epochs):
        loss = train(model, train_inputs, train_label)
        print(f'epoch: {i} loss: {loss}')

    accuracy = test(model, test_inputs, test_labels)
    print(f'model accuracy: {accuracy}')
    pass






