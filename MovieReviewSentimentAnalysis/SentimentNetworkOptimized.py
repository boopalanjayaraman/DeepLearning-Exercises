#This code originally appeared in a to-do exercise from Andrew Trask's session (Udacity, Deep Learning Nanodegree)
#Parts of the implementation were done by this author (Boopalan)
''' This code has been improved for optimizations for Faster training and testing by avoiding multiple matrix multiplications in the first step'''
 
import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        review_vocab = set()
        # TODO: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words 
        #       using "split(' ')" instead of "split()".
        for review in reviews:
            for word in review.split(' '):
                if(word != ' '):
                    review_vocab.add(word)
        
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        label_vocab = set()
        # TODO: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        for label in labels:
            label_vocab.add(label)
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i,word in enumerate(review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        # TODO: do the same thing you did for self.word2index and self.review_vocab, 
        #       but for self.label2index and self.label_vocab instead
        for i,word in enumerate(label_vocab):
            self.label2index[word] = i
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        
        # TODO: initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros([self.hidden_nodes, self.input_nodes])
        
        # TODO: initialize self.weights_1_2 as a matrix of random values. 
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes))
        
        # TODO: Create the input layer, a two-dimensional matrix with shape 
        #       1 x input_nodes, with all values initialized to zero
        
        #self.layer_0 = np.zeros((1,input_nodes))
        self.layer_1 = np.zeros((1, self.hidden_nodes))
    
        
    '''def update_input_layer(self,review):
        # TODO: You can copy most of the code you wrote for update_input_layer 
        #       earlier in this notebook. 
        #
        #       However, MAKE SURE YOU CHANGE ALL VARIABLES TO REFERENCE
        #       THE VERSIONS STORED IN THIS OBJECT, NOT THE GLOBAL OBJECTS.
        #       For example, replace "layer_0 *= 0" with "self.layer_0 *= 0"
        self.layer_0 *= 0
        
        word_counter = Counter()
        for word in review.split(' '):
            if word != ' ':
                #word_counter[word] += 1
                word_counter[word] = 1 #to reduce noise
        
        for word in word_counter:
            if(word in self.word2index):
                index = self.word2index[word]
                count = word_counter[word]
                self.layer_0[0, index] = count'''
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        #return self.label2index[label]
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1/(1+np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        return output * (1-output)

    def train(self, training_reviews_raw, training_labels):
        
        training_reviews = list()
        
        for review in training_reviews_raw:
            indices = set()    
            for word in review.split(' '):
                if word in self.word2index.keys(): 
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
        
        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
            # TODO: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            
            #self.update_input_layer(review) # input is self.layer_0
            '''self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1.T[index]
                
            hidden_inputs = self.layer_1.T
                '''
            
            target = np.array(self.get_target_for_label(label), ndmin=2).T # output 
            
            #hidden_inputs = np.dot(self.weights_0_1, self.layer_0.T) # shape --> h*i, i*1 = h*1
            hidden_inputs = np.zeros((1, self.hidden_nodes))
            for index in review:
                hidden_inputs += self.weights_0_1.T[index]
            
            #hidden_outputs = self.sigmoid(hidden_inputs) # shape --> h*1
            hidden_outputs = hidden_inputs.T #self.sigmoid(hidden_inputs) # shape --> h*1
            
            final_inputs = np.dot(self.weights_1_2, hidden_outputs) # shape --> o*h, h*1 = o*1
            final_outputs = self.sigmoid(final_inputs) # shape --> o*1
            
            # TODO: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
            output_errors = target - final_outputs
            output_error_grad = output_errors * self.sigmoid_output_2_derivative(final_outputs) # shape --> o*1
            
            hidden_errors = np.dot(output_error_grad.T, self.weights_1_2) # shape --> o*1, o*h = 1*h
            #hidden_error_grad = hidden_errors.T * self.sigmoid_output_2_derivative(hidden_outputs) #shape --> T(1*h), h*1 = 1*1 
            hidden_error_grad = hidden_errors.T # removing non-linearity? how? 
             
            #TODO: Update weights
            self.weights_1_2 += self.learning_rate * np.dot(output_error_grad, hidden_outputs.T) #shape --> o*1, T(h*1) = o*h
            #self.weights_0_1 += self.learning_rate * np.dot(hidden_error_grad, self.layer_0) #shape --> [1*1], [1*i] = 1 * i??
            #for h in range(self.hidden_nodes):
            for index in review:
                self.weights_0_1[:, index] += self.learning_rate * hidden_error_grad[0]
            
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            #if(np.abs(output_errors) < 0.5):
            #    correct_so_far +=1
            if(final_outputs >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(final_outputs < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review_raw):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        
        #self.update_input_layer(review.lower()) # input is self.layer_0
        
        indices = set()    
        for word in review_raw.split(' '):
            if (word != ' ') & (word in self.word2index): 
                indices.add(self.word2index[word])
        
        review = list(indices)

        #hidden_inputs = np.dot(self.weights_0_1, self.layer_0.T) # shape --> h*i, i*1 = h*1
        hidden_inputs = np.zeros((1, self.hidden_nodes))
        for index in review:
            hidden_inputs += self.weights_0_1.T[index]
        
        '''self.layer_1 *= 0
        for index in review:
            self.layer_1 += self.weights_0_1.T[index]
        
        hidden_inputs = self.layer_1.T'''
        
        #hidden_outputs = self.sigmoid(hidden_inputs) # shape --> h*1
        hidden_outputs = hidden_inputs.T # shape --> h*1

        final_inputs = np.dot(self.weights_1_2, hidden_outputs) # shape --> o*h, h*1 = o*1
        final_outputs = self.sigmoid(final_inputs) # shape --> o*1
        
        # TODO: The output layer should now contain a prediction. 
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`, 
        #       and `NEGATIVE` otherwise.
        if np.abs(final_outputs[0]) >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

def Run():
	mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)
	#train the model
	mlp.train(reviews[:-1000],labels[:-1000])
	#test now with test set
	mlp.test(reviews[-1000:],labels[-1000:])

			
if __name__=='__main__':
	Run()