import time
import tensorflow as tf
from tensorflow.keras import activations
import numpy as np
from preprocess import get_data, TARGET_HEIGHT, TARGET_WIDTH

class Art_Model(tf.keras.Model):
    def __init__(self, num_classes, width, height):
        super(Art_Model, self).__init__()

        #hyperparameters:
        self.batch_size = 100
        self.num_classes = num_classes
        self.width = width
        self.height = height

        self.hidden_dim = 256
        self.learning_rate = .001
        self.epochs = 10

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        input_shape = (self.width, self.height, 3) #to account for different shaped images
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Conv2D(filters=8, 
                kernel_size=4,
                activation='relu',
                input_shape=input_shape,
                padding='same'))
        self.classifier.add(tf.keras.layers.Dense(self.hidden_dim))
        self.classifier.add(tf.keras.layers.LeakyReLU(alpha=.3))
        self.classifier.add(tf.keras.layers.Dropout(.25))
        self.classifier.add(tf.keras.layers.Dense(self.hidden_dim))
        self.classifier.add(tf.keras.layers.LeakyReLU(alpha=.3))
        self.classifier.add(tf.keras.layers.Flatten())
        self.classifier.add(tf.keras.layers.Dropout(.25))
        self.classifier.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        # things maybe to tweak:
        # - add maxpool
        # - tweak width & depth of convo layers

    def call(self, inputs):
        """Performs forward pass through Art_Model by applying convolution
        and dense layers to inputs.
        
        Inputs:
        -inputs: Batch of input images of shape (N, H, W, 3)
        
        Returns:
        -logits: probability distribution over all the labels for each image
        of shape (N, C)"""
        logits = self.classifier(inputs)
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
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
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
        inputs = shuffled_inputs[i*model.batch_size: (i+1)*model.batch_size]
        #inputs = tf.image.random_flip_left_right(inputs)
        
        labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

        with tf.GradientTape() as gt:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)
            # model.loss_list.append(loss)

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
    ind = tf.random.shuffle(np.array(range(0, len(test_inputs))))
    shuffled_inputs = tf.gather(test_inputs, ind)
    shuffled_labels = tf.gather(test_labels, ind)

    total_accuracy = 0.0

    num_batches = int(len(test_inputs)/model.batch_size)
    
    for i in range(0, num_batches):      
        inputs = shuffled_inputs[i*model.batch_size: (i+1)*model.batch_size]
        #inputs = tf.image.random_flip_left_right(inputs)
        
        labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

        logits = model.call(inputs)
        accuracy = model.accuracy(logits, labels)

        total_accuracy += accuracy
    
    return total_accuracy / num_batches
    logits = model.call(test_inputs)
    print("A: ", logits.size)
    print("B: ", test_labels.size)
    return
    test_accuracy = model.accuracy(logits, test_labels)
    # print("C: ", test_accuracy.shape)

    return test_accuracy

def main():
    """ 
    Reads in image data (12 classes), initializes Art_Model, and trains model for 
    model.epochs epochs.

    Prints:
    - loss after each epoch
    - model accuracy at end of training 
    """

    preprocess_start = time.time()
    print("PREPROCESSING...")
    train_inputs, train_labels, test_inputs, test_labels = get_data('data')
    print("preprocess finished in ", time.time() - preprocess_start)
    print("---------------------------")

    train_start = time.time()
    model=Art_Model(num_classes=12, width=TARGET_WIDTH, height=TARGET_HEIGHT)
    print("TRAINING MODEL...")
    for i in range(model.epochs):
        epoch_start = time.time()
        loss = train(model, train_inputs, train_labels)
        print("---- epoch finished in ", time.time() - epoch_start)
        print(f'epoch: {i+1} loss: {loss}')
    print("training finished in ", time.time() - train_start)
    print("---------------------------")

    test_start = time.time()
    print("TESTING MODEL...")
    accuracy = test(model, test_inputs, test_labels)
    print(accuracy)
    print("testing finished in ", time.time() - test_start)
    # print(f'model accuracy: {accuracy}')
    print("---------------------------")

if __name__ == '__main__':
	main()
