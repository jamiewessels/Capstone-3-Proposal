from tensorflow.keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model
import os
import os.path
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_dir = os.path.join(os.curdir, 'CovNet_logs')  

# make generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                fill_mode='constant', height_shift_range=0.05, width_shift_range=0.05,
                rotation_range=1, zoom_range = 0.05)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


def create_transfer_model(base_model, n_categories=5):
        '''
        Takes in a base model and returns model with softmax output with n_categories neurons 

                Parameters:
                        base_model with loaded weights
                        n_categories (int): number of output neurons
                Returns:
                        model
        '''    
        avg = GlobalAveragePooling2D()(base_model.output) 
        output = Dense(n_categories, activation='softmax')(avg)
        model = Model(inputs=base_model.input, outputs=output)
        
        return model

def print_model_properties(model, indices = 0):
        '''
        Takes in a model and prints each layer's name and whether layer is trainable 

                Parameters:
                        loaded model
                        indices (int): starting layer index
                Returns:
                        print statement of each layer's name and whether layer is trainable
        '''    
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

def change_trainable_layers(model, trainable_index):
        '''
        Changes layers to trainable (unfreeze)

                Parameters:
                        loaded model
                        trainable_index(int): index at and above which layers will be trainable (unfrozen)
                Returns:
                        model 
        '''    
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

def get_tboard_logdir():
        '''
        Returns:
                name for tensorboard log
        '''    
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M")
        return os.path.join(root_dir, run_id)

def score_model(model, test_generator, num_test):
        '''
        Scores model. Assumes batch size of 32

                Parameters:
                        model: loaded model with loaded weights
                        test_generator: preprocessed test image generator
                        num_test (int): number of images in test_generator
                Returns:
                        dictionary with scoring metrics
        '''    
        results_dict = model.evaluate(test_generator, steps = num_test//32, return_dict = True)
        ytrue = test_generator.classes
        probas = model.predict(test_generator)
        yhat = np.argmax(probas, axis = 1)
        results_dict['recall'] = recall_score(ytrue, yhat, average = 'macro')
        results_dict['precision'] = precision_score(ytrue, yhat, average = 'macro')
        return results_dict

def get_confusion_matrix(model, test_generator):
        '''
        Creates confusion matrix for test samples.

                Parameters:
                        model: loaded model with loaded weights
                        test_generator: preprocessed test image generator
                Returns:
                        confusion matrix
        '''    

        return confusion_matrix(test_generator.classes, np.argmax(model.predict(test_generator), axis = 1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        '''
        Plots confusion matrix. 

                Parameters:
                        cm: confusion matrix from get_confusion_matrix function
                        classes (list): labels for classes
                        normalize (bool): if True, entries will be percentages instead of counts
                        title (str): title for confusion matrix 
                Returns:
                        plotted confusion matrix
        '''    

        import itertools
        if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
        else:
                print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('images/conf_matrix_5chords_final') 

if __name__ == '__main__':

        num_train = len([os.path.join(path, name) for path, subdirs, files in os.walk('data_5chords/train/') for name in files])+1
        num_val = len([os.path.join(path, name) for path, subdirs, files in os.walk('data_5chords/val') for name in files])+1
        num_test = len([os.path.join(path, name) for path, subdirs, files in os.walk('data_5chords/test') for name in files])+1

        #Vars
        batch_size = 32
        target_size = (299, 299)
        steps_per_epoch = num_train // batch_size
        validation_steps = num_val // batch_size

        # make flow from directory objects
        train_generator = train_datagen.flow_from_directory(
                'data_5chords/train/',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical')

        validation_generator = valid_datagen.flow_from_directory(
                'data_5chords/val',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical', 
                shuffle = False)

        test_generator = test_datagen.flow_from_directory(
                'data_5chords/test',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical', 
                shuffle = False)

        # instantiate callbacks
        run_tboard_dir = get_tboard_logdir()
        tensorboard = TensorBoard(log_dir=run_tboard_dir, histogram_freq=2, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        
        #Transfer Model was Created from Best Weights with 3 Chord Classifier
        '''
        weights = load_model('./CovNet_logs/best_model_3chords').get_weights()[:-2]
        base_model = Xception(weights= 'imagenet',
                        include_top=False,
                        input_shape=(299,299,3))

        base_model.set_weights(weights)
        
        transfer_model = create_transfer_model(base_model, n_categories=5)
        '''

        best_model = load_model('./CovNet_logs/best_model_5chords.hdf5')

        metrics = score_model(best_model, validation_generator, num_val)
        cm = get_confusion_matrix(best_model, validation_generator)
        plot_confusion_matrix(cm, ['Am', 'C', 'Dm', 'F', 'G'], normalize = True)
