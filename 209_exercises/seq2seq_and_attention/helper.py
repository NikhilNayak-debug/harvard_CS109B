# Import necessary libraries
import tensorflow as tf 
import numpy as np
import pandas as pd


def get_dataset(token_eng, token_spa):
	# Get a list of all tokens in English
	tokens_ = [j for i in token_eng for j in i]

	# Find the number of tokens
	N_words_eng = len(set(tokens_))

	# To compute the loss of the decoder, the start token is not required for all Spanish inputs
	target_spa = [i[1:] for i in token_spa]

	# To give as an input to the decoder we only need the start <s> token
	# The end </s> token is not required for all Spanish inputs
	token_spa = [i[:-1] for i in token_spa]


	# Create a ragged tensor from the token lists

	# Ragged tensors are the TensorFlow equivalent of nested variable-length lists. 
	# They make it easy to store and process data with non-uniform shapes

	# Create ragged tensor constant for the English tokens which is to be given as the 
	# input to the encoder
	ragged_eng = tf.ragged.constant(token_eng)

	# Create ragged tensor constant for the Spanish tokens which is to be given as the 
	# input to the decoder
	ragged_spa = tf.ragged.constant(token_spa)

	# Create ragged tensor constant for the Spanish tokens created for computing the decoder loss
	ragged_target = tf.ragged.constant(target_spa)

	# Use from_tensor_slices to create a dataset of the form (x,y) where x is the English
	# tokens and y is the Spanish translation. Also add the tensor used for loss computation 
	# The dataset will be of the form - ((input_eng, input_spa), target_spa)
	dataset = tf.data.Dataset.from_tensor_slices((ragged_eng, ragged_spa, ragged_target))

	# Shuffle the dataset
	dataset = dataset.shuffle(5000)

	# Set the size of the batch
	dataset = dataset.batch(1024)

	# Use .map to transform the dataset
	# This transformation applies map_func to each element of this dataset, 
	# and returns a new dataset containing the transformed elements
	dataset = dataset.map(lambda x,y,z: ((x.to_tensor(default_value=0, shape=[None, None]), y.to_tensor(default_value=0, shape=[None, None])), z.to_tensor(default_value=0, shape=[None, None])) , num_parallel_calls=5)

	# Creates a Dataset that prefetches elements from this dataset.
	# Most dataset input pipelines should end with a call to prefetch. 
	# This allows later elements to be prepared while the current element is being processed.
	# This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
	dataset = dataset.prefetch(1)

	return dataset




