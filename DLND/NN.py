import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])
	

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer, 2x3, 3x1 = 2x1
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer, 
        
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error - Replace this value with your calculations.
        output_errors = targets - final_outputs # Output layer error is the difference between desired target and actual output.
        output_error_grad =  output_errors * 1 #final_outputs * (1-final_outputs) #1x1
        
        # TODO: Backpropagated error - Replace these values with your calculations.
        hidden_errors = np.dot(output_error_grad, self.weights_hidden_to_output)  # errors propagated to the hidden layer, 1x1 . 1x2 = 1x2
        hidden_grad = hidden_errors.T * hidden_outputs * (1 - hidden_outputs) # hidden layer gradients, T[2x1].[2x1] = 1x1
        
        '''print(hidden_outputs.shape)
        print(hidden_errors.shape)
        print(hidden_grad.shape)
        print(self.weights_input_to_hidden.shape)
        print(inputs.shape)'''
    
        
        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * output_error_grad * hidden_outputs.T # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_grad *  inputs.T # update input-to-hidden weights with gradient descent step
 
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs
		
def Run():	
	network = NeuralNetwork(3, 2, 1, 0.5)
	print('Train Result:')
	network.weights_input_to_hidden = test_w_i_h.copy()
	network.weights_hidden_to_output = test_w_h_o.copy()
	
	network.train(inputs, targets)
	print(network.weights_hidden_to_output)
	print(network.weights_input_to_hidden)
	print(np.allclose(network.weights_hidden_to_output, 
								np.array([[ 0.37275328, -0.03172939]])))
	print(np.allclose(network.weights_input_to_hidden,
								np.array([[ 0.10562014,  0.39775194, -0.29887597],[-0.20185996,  0.50074398,  0.19962801]])))
	print('Expected Values:')
	print(np.array([[ 0.37275328, -0.03172939]]))
	print(np.array([[ 0.10562014,  0.39775194, -0.29887597],[-0.20185996,  0.50074398,  0.19962801]]))
	
	print('Run Result:')
	network.weights_input_to_hidden = test_w_i_h.copy()
	network.weights_hidden_to_output = test_w_h_o.copy()
	
	print(network.run(inputs))
	print(np.allclose(network.run(inputs), 0.09998924))
	print('Expected value:')
	print(0.09998924)

	
if __name__=='__main__':
	Run()