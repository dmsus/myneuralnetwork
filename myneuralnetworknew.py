#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#defining class of neural network
class neuralNetwork:
    
    #initializing neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # create quantity of nodes on each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #learning rate
        self.lr = learningrate
        
        #matrix weighted wih - between input and hidden
        # who - between hidden and output
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        #sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    #training neural network
    def train(self, inputs_list, targets_list):
        #transform list of input to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #count input signal for hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #count output signal for hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #count input signal for final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #count output signal for final layer
        final_outputs = self.activation_function(hidden_inputs)
        
        #errors of input layer
        output_errors = targets - final_outputs
        #hidden errors
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #update date between hidden and outputs layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        #update date between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    #checking neural network
    def query(self, inputs_list):
        #transfer list of input to array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #count input signal for hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        #count output signal for hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #count input signal for final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #count output signal for final layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    


# In[ ]:


#create example of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[ ]:


#test
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# In[ ]:


#load text file
training_data_file = open("C:\\Users\\cutyd\\OneDrive\\Desktop\\IT\\MNIST\\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[ ]:



epochs = 5
for e in range(epochs):
    # training neural network
    # transform data to training type list
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] target marker value for this
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[ ]:


test_data_file = open("C:\\Users\\cutyd\\OneDrive\\Desktop\\IT\\MNIST\\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[ ]:


#testing neural network
scorecard = []

for records in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0]) 
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass


# In[ ]:


scorecard_array = numpy.asarray(scorecard)
print("efficiency = ", scorecard_array.sum() / scorecard_array.size)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




