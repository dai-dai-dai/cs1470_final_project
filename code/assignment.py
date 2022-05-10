import time
import tensorflow as tf
from tensorflow.keras import activations
import numpy as np
import csv
from preprocess import get_data, TARGET_HEIGHT, TARGET_WIDTH, GENRE_TO_INDEX, INDEX_TO_GENRE

class Art_Model(tf.keras.Model):
    def __init__(self, num_classes, width, height):
        super(Art_Model, self).__init__()

        # hyperparameters
        self.batch_size = 350
        self.num_classes = num_classes
        self.width = width
        self.height = height
        self.hidden_dim = 412
        self.learning_rate = .001
        self.epochs = 10

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # sequential model
        input_shape = (self.width, self.height, 3) # to account for different shaped images
        self.classifier = tf.keras.Sequential()
        
        self.classifier.add(tf.keras.layers.Conv2D(filters=8, 
                kernel_size=4,
                activation='relu',
                input_shape=input_shape,
                padding='same'))
        self.classifier.add(tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=.001, center=True))
        self.classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.classifier.add(tf.keras.layers.Conv2D(filters=8, 
                kernel_size=4,
                activation='relu',
                input_shape=input_shape,
                padding='same'))
        self.classifier.add(tf.keras.layers.BatchNormalization(
            axis=-1, epsilon=.001, center=True))
        self.classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        self.classifier.add(tf.keras.layers.Dense(self.hidden_dim))
        self.classifier.add(tf.keras.layers.LeakyReLU(alpha=.3))
        self.classifier.add(tf.keras.layers.Dropout(.25))
        self.classifier.add(tf.keras.layers.Dense(self.hidden_dim))
        self.classifier.add(tf.keras.layers.LeakyReLU(alpha=.3))
        self.classifier.add(tf.keras.layers.Flatten())
        self.classifier.add(tf.keras.layers.Dropout(.25))
        self.classifier.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

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


# -------------------- TRAIN / TEST --------------------

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
        
        labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

        with tf.GradientTape() as gt:
            logits = model.call(inputs)
            loss = model.loss(logits, labels)

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

    num_batches = max(1, int(len(test_inputs)/model.batch_size))
    
    for i in range(0, num_batches):      
        inputs = shuffled_inputs[i*model.batch_size: (i+1)*model.batch_size]
        
        labels = shuffled_labels[i*model.batch_size: (i+1)*model.batch_size]

        logits = model.call(inputs)
        accuracy = model.accuracy(logits, labels)

        total_accuracy += accuracy
    
    return total_accuracy / num_batches


# -------------------- METRIC FUNCTIONS --------------------

def sample(model, sample_files, sample_images, sample_labels):
    """
    Runs the model on randomly sampled images
    Inputs: 
    - model: instance of Art_Model
    - sample_files: at each index, contains NUM_SAMPLES random filepaths sampled from the genre index
    - sample_images: at each index, contains corresponding image tensors
    - sample_labels: at each index, contains corresponding label tensors

    Returns:
    - rows: csv rows containing file name, true label, predicted label, and per image prediction distribution
    """
    rows = [["filename", "true label", "predicted", "abstract_expressionism", "baroque", "cubism", "fauvism", "high_renaissance", "iconoclasm", "impressionism", "old_greek_pottery", "realism", "rococo", "romanticism", "surrealism"]]
    for i in range(len(sample_images)):
        for j in range(len(sample_images[i])):
            filepath = sample_files[i][j]
            genre = filepath.split('/')[1]
            image = tf.reshape(sample_images[i][j], shape=(1, TARGET_HEIGHT, TARGET_WIDTH, 3))
            logit = model.call(image)
            predicted = tf.get_static_value(tf.argmax(logit, 1))[0]
            row = [filepath, genre, INDEX_TO_GENRE[predicted]]
            for prob in logit[0]:
                row.append(str(tf.get_static_value(prob)))
            rows.append(row)
    return rows

def get_categorical_accuracy(model, images_by_category, matching_labels):
    """
    Runs the model on images by category to compute categorical accuracy and overall prediction distribution
    per genre
    Inputs: 
    - model: instance of Art_Model
    - images_by_category: at each index, contains all image tensors of the genre index
    - matching_labels: at each index, contains corresponding label tensors

    Returns:
    - rows: csv rows containing file name, true label, predicted label, and per image prediction distribution
    """
    fields = ["genre", "categorical accuracy", "abstract_expressionism", "baroque", "cubism", "fauvism", "high_renaissance", "iconoclasm", "impressionism", "old_greek_pottery", "realism", "rococo", "romanticism", "surrealism"]
    csv_dict = []
    categorical_accuracy = []
    prediction_distribution = []
    # per painting genre
    for i in range(model.num_classes):
        # calculate prediction distribution
        images = images_by_category[i]
        ind = tf.random.shuffle(np.array(range(0, len(images))))
        shuffled_inputs = tf.gather(images, ind)
        logits = model.call(shuffled_inputs)
        y, _, counts = tf.unique_with_counts(tf.argmax(logits, 1))
        prediction_distribution.append((y, counts))
        # calculate categorical accuracy
        accuracy = test(model, images_by_category[i], matching_labels[i])
        categorical_accuracy.append(tf.get_static_value(accuracy))
    # construct csv rows
    for i in range(model.num_classes):
        row = {"abstract_expressionism":0, "baroque":0, "cubism":0, "fauvism":0, "high_renaissance":0, "iconoclasm":0, "impressionism":0, "old_greek_pottery":0, "realism":0, "rococo":0, "romanticism":0, "surrealism":0}
        row["genre"] = INDEX_TO_GENRE[i]
        row["categorical accuracy"] = str(categorical_accuracy[i])
        predictions = prediction_distribution[i]
        y, counts = predictions[0], predictions[1]
        total = tf.reduce_sum(counts)
        for j in range(len(y)):
            genre = INDEX_TO_GENRE[tf.get_static_value(y[j])]
            row[genre] = str(tf.get_static_value(counts[j] / total))
        csv_dict.append(row)
    return fields, csv_dict


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
    train_inputs, train_labels, test_inputs, test_labels, sample_files, sample_images, sample_labels, images_by_category, matching_labels = get_data('data')
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
    print("testing finished in ", time.time() - test_start)
    print(f'model accuracy: {accuracy}')
    print("---------------------------")

    sample_start = time.time()
    print("SAMPLING...")
    rows = sample(model, sample_files, sample_images, sample_labels)
    with open("sample_2.csv", 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)     
        csvwriter.writerows(rows) 
    print("sampling finished in ", time.time() - sample_start)
    print("---------------------------")

    cat_acc_start = time.time()
    print("CALCULATING METRICS...")
    fields, rows = get_categorical_accuracy(model, images_by_category, matching_labels)
    with open("categorical_metrics_2.csv", 'w') as csvfile:  
        csvwriter = csv.DictWriter(csvfile, fieldnames = fields)  
        csvwriter.writeheader()  
        csvwriter.writerows(rows) 
    print("metrics finished in ", time.time() - cat_acc_start)
    print("---------------------------")


if __name__ == '__main__':
	main()
