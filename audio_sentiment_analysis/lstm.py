import os
import csv
import time
import random
import numpy as np
import theano
import theano.tensor
import lasagne
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle
import argparse

# set random seed
np.random.seed(10)


def preprocess_data(feat_loc, max_length = 0):
    '''
    Preprocesses data so that it can be used by Lasagne network. Masks all inputs
    shorter than the max length with 0's so network can accept variable length
    sequence inputs. Returns shuffled masks, labels, and data.
    '''
    feature_file = pd.read_csv(feat_loc)
    ids = feature_file.ix[:, 0].values
    labels = feature_file.ix[:, -1].values
    data = feature_file.ix[:, 1:-1].values

    # find what the max length should be
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(data, labels, ids):
        if len(test_index) > max_length:
            max_length = len(test_index)
            MAX_LENGTH = max_length

    # loop through again to mask inputs with 0's that are shorter than the max
    inputs = []
    masks = []
    targets = []
    for train_index, test_index in logo.split(data, labels, ids):
        # test data
        curr_data = data[test_index]
        masks.append([1 if x < len(test_index) else 0 for x in range(max_length)])
        inputs.append([curr_data[x] if x < len(test_index) else [0]*len(curr_data[0]) for x in range(max_length)])
        targets.append(labels[test_index][0])

    inputs = np.array(inputs)
    masks = np.array(masks)
    targets = np.array(targets).reshape(len(targets), 1)

    # shuffle the array inputs
    inputs, masks, targets = shuffle(inputs, masks, targets, random_state=10)

    # also return the number of features for building the network
    return inputs, masks, targets, data.shape[1]


def build_network(input_var=None, mask_var=None, num_features=0, num_units=10):
    '''
    Creates the bidirectional LSTM. 
    '''
    # Input layer - shape = (batch size, max sequence length, number of features)
    layer_in = lasagne.layers.InputLayer(shape=(None, None, num_features), input_var=input_var)
    
    # Masked input layer
    layer_mask = lasagne.layers.InputLayer((None, None), mask_var)
    
    # Hidden forward LSTM layer
    layer_forward = lasagne.layers.LSTMLayer(layer_in, 
                                             num_units = num_units,
                                             mask_input = layer_mask, 
                                             backwards=False, 
                                             peepholes=True,
                                             only_return_final=True)
    
    # Hidden backwards LSTM layer
    layer_backward = lasagne.layers.LSTMLayer(layer_in,
                                              num_units = num_units,
                                              mask_input = layer_mask, 
                                              backwards=True, 
                                              peepholes=True,
                                              only_return_final=True)
    
    # concatenates the forwards and backwards layers
    layer_concat = lasagne.layers.ConcatLayer([layer_forward,layer_backward])

    layer_out = lasagne.layers.DenseLayer(layer_concat, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    

    return layer_out


def main(args, pipe=False):
    '''
    Checks passed arguments and performs requested actions.
    '''
    if not pipe:
        parser = argparse.ArgumentParser(description='Classify calls as positive or negative.')
        parser.add_argument('-f', '--features', dest='feat_loc', required=True,
                            help='Path to CSV feature file.')
        parser.add_argument('-o', '--out', dest='out_loc', required=True,
                            help='Path to where classification summary should be saved.')
        parser.add_argument('--epochs', dest='num_epochs', help='Number of training epochs.')
        parser.add_argument('--n_units', dest='num_units', help='Number of LSTM units.')
        args = parser.parse_args()

    # grab arguments if given, otherwise give some defaults
    if args.num_epochs:
        num_epochs = int(args.num_epochs)
    else:
        num_epochs = 15
    if args.num_units:
        num_units = int(args.num_units)
    else:
        num_units = 10

    # Prepare data and set some parameters
    inputs, masks, targets, num_features = preprocess_data(args.feat_loc)

    # Build neural network model 
    print("Building network ...")
    
    # Prepare Theano variables for targets
    input_var = theano.tensor.tensor3('inputs')
    target_var = theano.tensor.imatrix('targets')
    mask_var = theano.tensor.matrix('mask')

    # Initialize neural network model
    network = build_network(input_var, mask_var, num_features=num_features, num_units=num_units)

    # Save initial params for resetting after each fold
    original_params = lasagne.layers.get_all_param_values(network)

    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # Create update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)
    
    # Create a loss expression for validation/testing. 
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    
    # Create an accuracy expression for validation/testing
    test_acc = lasagne.objectives.binary_accuracy(test_prediction,target_var)
    test_acc = test_acc.mean()

    # Compiling functions
    print("Compiling functions ...")
    
    # Compile a function performing a training step 
    train_fn = theano.function([input_var, target_var, mask_var], loss, updates=updates, allow_input_downcast=True)
    
    # Compile the validation loss and accuracy function
    val_fn = theano.function([input_var, target_var, mask_var], [test_loss, test_acc], allow_input_downcast=True)

    # Training loop
    print("Starting training...\n")
    
    # Iteration over epochs
    try:
        cum_acc = []
        cum_train_acc = []
        split_count = 1
        for test_index in range(len(targets)):
            test_data = inputs[test_index].reshape((-1, inputs[test_index].shape[0], inputs[test_index].shape[1]))
            train_data = inputs[np.arange(len(targets))!=test_index]
            test_masks = masks[test_index].reshape((-1, masks[test_index].shape[0]))
            train_masks = masks[np.arange(len(targets))!=test_index]
            test_targets = targets[test_index].reshape((-1, targets[test_index].shape[0]))
            train_targets = targets[np.arange(len(targets))!=test_index]

            # reset weights after each fold
            lasagne.layers.set_all_param_values(network, original_params)

            print("Epoch\t|\tTime\t|    Training loss\t|    Validation loss\t| Validation Accuracy")
            print("_____________________________________________________________________________________________")

            for epoch in range(num_epochs):
                
                # Start timer
                start_time = time.time()
                
                # Full pass over the training data
                train_err = train_fn(train_data, train_targets, train_masks)
            
                # Full pass over the validation data
                val_err, val_acc =  val_fn(test_data, test_targets, test_masks)
                    
                # Print the results for this epoch
                print("{}\t|\t{:.3f}\t|\t{:.6f}\t|\t{:.6f}\t|\t{:.2f}\t".format(
                    epoch + 1,time.time() - start_time,
                    float(train_err),float(val_err),val_acc * 100))

                if split_count == 1:
                    losses.append(float(train_err))
            
            # Compute the test error
            test_err, test_acc = val_fn(test_data, test_targets, test_masks)
            train_err, train_acc =  val_fn(train_data, train_targets, train_masks)

            # get the output label
            if test_acc == 1.0:
                pred_label = test_targets[0][0]
            else:
                pred_label = 1 - test_targets[0][0]
        
            # Print test error values
            print("\nFinal results for split: " + str(split_count))
            print("  training loss:\t\t{:.6f}".format(float(train_err)))
            print("  validation loss:\t\t{:.6f}".format(float(val_err)))
            print("  test loss:\t\t\t{:.6f}".format(float(test_err)))
            print("  training accuracy:\t\t{:.2f} %".format(train_acc * 100))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100))
            print("  test accuracy:\t\t{:.2f} %".format(test_acc * 100))
            print("  Real label: " + str(test_targets[0][0]))
            print("  Predicted label: " + str(pred_label))

            # append test accuracy for calculating overall score
            cum_acc.append(test_acc)
            cum_train_acc.append(train_acc)

            split_count += 1

        # overall test accuracy for all splits
        with open(os.path.join(args.out_loc, 'deep-results.txt'), 'a') as results:
            results.write('Epochs: ' + str(num_epochs) + ' Units: ' + str(num_units) + '\n')
            results.write('Overall test accuracy: ' + str(np.mean(cum_acc) * 100) + '\n')
            results.write('Overall train accuracy: ' + str(np.mean(cum_train_acc) * 100) + '\n')        

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main(sys.argv[1:])