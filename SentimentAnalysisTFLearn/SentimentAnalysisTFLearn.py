'''
Sections of this code was originally done as part of the Udacity course - Deep learning Nanodegree
Parts of the code were implemented by the current author
'''
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical


def execute():
	
	##prepare the data by loading
	reviews = pd.read_csv('reviews.txt', header=None)
	labels = pd.read_csv('labels.txt', header=None)
	
	##count all the words and prepare the vocabulary
	from collections import Counter
	total_counts = Counter()# bag of words here

	for idx,row in reviews.iterrows():
		words = row[0].split(' ')
		total_counts.update(words)

	print("Total words in data set: ", len(total_counts))
	
	##considering only frequently used 10000 words for learning. (can also be achieved by using most_common method of counter)
	vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
	print(vocab[:60])
	
	##creating the word to index dictionary lookup here. This will be used while preparing the input data
	word2idx = {}  
	for index, word in enumerate(vocab):
		word2idx[word] = index
	
	##convert all reviews into input vectors - so that they can be split and used for both training and testing.
	word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
	for ii, (_, text) in enumerate(reviews.iterrows()):
		word_vectors[ii] = text_to_vector(text[0])
	
	##prepare training and testing sets, and prepare target data
	Y = (labels=='positive').astype(np.int_)
	records = len(labels)

	shuffle = np.arange(records)
	np.random.shuffle(shuffle)
	test_fraction = 0.9

	train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
	trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split, 0], 2)
	testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split, 0], 2)

	##initialize model
	model = build_model()
	##train the model
	model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=50)
	
	##test the model
	predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
	test_accuracy = np.mean(predictions == testY[:,0], axis=0)
	

def build_model():
        
    ##since input data nodes are 10000 here (as per vocab), we'll need 10000 input nodes
    net = tflearn.input_data([None, 10000])
    #net = tflearn.fully_connected(net, 5000, activation = 'ReLU') #used earlier, the network took longer time to initialize and run
    net = tflearn.fully_connected(net, 200, activation = 'ReLU')
    net = tflearn.fully_connected(net, 25, activation = 'ReLU')
    net = tflearn.fully_connected(net, 2, activation = 'softmax')
    net = tflearn.regression(net, optimizer = 'sgd', learning_rate=0.1, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model
	
'''This function will take the review text as input and return the corresponding input vector as output'''
def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)		

def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} :'.format(positive_prob), 
          'Positive' if positive_prob > 0.5 else 'Negative')
	
if __name__ == '__main__':
	execute()
	sentence = "Moonlight is by far the best movie of 2016."
	test_sentence(sentence)

	sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
	test_sentence(sentence)

	sentence = "Bahubali 2 Blu-Ray Rocks.....Superb Sound Effect....Amazing Picture Quality.....Just Go for this Blu Ray"
	test_sentence(sentence)