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
from sklearn.metrics import recall_score, precision_score, roc_auc_score
import numpy as np
import pandas as pd

root_dir = os.path.join(os.curdir, 'CovNet_logs')  

# make generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
                fill_mode='constant', height_shift_range=0.05, width_shift_range=0.05,
                rotation_range=1, zoom_range = 0.05)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


def create_transfer_model(base_model, n_categories=5):
        
        # add new head
        avg = GlobalAveragePooling2D()(base_model.output)
        output = Dense(n_categories, activation='softmax')(avg)
        model = Model(inputs=base_model.input, outputs=output)
        
        return model

def print_model_properties(model, indices = 0):
     for i, layer in enumerate(model.layers[indices:]):
        print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

def change_trainable_layers(model, trainable_index):
    for layer in model.layers[:trainable_index]:
        layer.trainable = False
    for layer in model.layers[trainable_index:]:
        layer.trainable = True

def get_tboard_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M")
        return os.path.join(root_dir, run_id)

def score_model(model, test_generator, num_test):
        results_dict = model.evaluate(test_generator, steps = num_test//32, return_dict = True)
        ytrue = test_generator.classes
        probas = best_model.predict(test_generator)
        yhat = np.argmax(probas, axis = 1)
        results_dict['recall'] = recall_score(ytrue, yhat, average = 'macro')
        results_dict['precision'] = precision_score(ytrue, yhat, average = 'macro')
        return results_dict 
        

if __name__ == '__main__':

        num_train = len([os.path.join(path, name) for path, subdirs, files in os.walk('data/train/augmented_2') for name in files])+1
        num_valid = len([os.path.join(path, name) for path, subdirs, files in os.walk('data/val') for name in files])+1
        num_test = len([os.path.join(path, name) for path, subdirs, files in os.walk('data/test') for name in files])+1

        #Vars
        batch_size = 32
        target_size = (299, 299)
        steps_per_epoch = num_train // batch_size
        validation_steps = num_valid // batch_size

        # make flow objects
        train_generator = train_datagen.flow_from_directory(
                'data/train/augmented_2',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical')

        validation_generator = valid_datagen.flow_from_directory(
                'data/val',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical')

        test_generator = test_datagen.flow_from_directory(
                'data/test',
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical', 
                shuffle = False)

        # instantiate callbacks
        run_tboard_dir = get_tboard_logdir()
        tensorboard = TensorBoard(log_dir=run_tboard_dir, histogram_freq=2, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        
        #Create transfer model based on weights from 3 classes (checkpoint6)
        ''' 
        weights = load_model('./CovNet_logs/Checkpoint6.hdf5').get_weights()[:-2]
        base_model = Xception(weights= 'imagenet',
                        include_top=False,
                        input_shape=(299,299,3))

        base_model.set_weights(weights)
        
        transfer_model = create_transfer_model(base_model, n_categories=5)

        #First round, just change weights for the output layer
        _ = change_trainable_layers(transfer_model, 132)
        print_model_properties(transfer_model)

        #Compile and fit first model - just outer layer tuning
        optimizer = SGD(lr=0.2, momentum = 0.9, decay = 0.01)
        model_cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='./CovNet_logs/Checkpoint1B.hdf5')
        transfer_model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy', 'AUC'])

        history = transfer_model.fit(x=train_generator, 
                        validation_data=validation_generator,
                        epochs=50, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps,
                        callbacks=[tensorboard, early_stopping, model_cp])
        '''
        

        
        transfer_model_2 = load_model('./CovNet_logs/Checkpoint1B.hdf5')

        _ = change_trainable_layers(transfer_model_2, 126) 

        print_model_properties(transfer_model_2)

        optimizer = SGD(lr=0.01, momentum = 0.9, decay = 0.001)
        transfer_model_2.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy', 'AUC'])
        model_cp2 = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='./CovNet_logs/Checkpoint2B.hdf5')

        history = transfer_model_2.fit(x=train_generator, 
                        validation_data=validation_generator,
                        epochs=500, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps,
                        callbacks=[model_cp2, early_stopping, tensorboard])



        '''
        
        transfer_model_3 = load_model('./CovNet_logs/Checkpoint8.hdf5')

        model_cp3 = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='./CovNet_logs/Checkpoint9.hdf5')

        _ = change_trainable_layers(transfer_model_3, 116) 

        print_model_properties(transfer_model_3)

        optimizer = SGD(lr=0.01, momentum = 0.9, decay = 0.001)
        transfer_model_3.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy', 'AUC'])

        history = transfer_model_3.fit(x=train_generator, 
                        validation_data=validation_generator,
                        epochs=500, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps,
                        callbacks=[model_cp3, early_stopping, tensorboard])
        
        
        

        transfer_model_4 = load_model('./CovNet_logs/Checkpoint9.hdf5')

        model_cp4 = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='./CovNet_logs/Checkpoint10.hdf5')

        _ = change_trainable_layers(transfer_model_4, 106) 

        print_model_properties(transfer_model_4)

        optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
        transfer_model_4.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy', 'AUC'])

        history = transfer_model_4.fit(x=train_generator, 
                        validation_data=validation_generator,
                        epochs=500, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps,
                        callbacks=[model_cp4, early_stopping, tensorboard])

        

        
        transfer_model_5 = load_model('./CovNet_logs/Checkpoint10.hdf5')

        model_cp5 = ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath='./CovNet_logs/Checkpoint11.hdf5')

        _ = change_trainable_layers(transfer_model_6, 96) 

        print_model_properties(transfer_model_5)

        optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
        transfer_model_5.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy', 'AUC'])

        history = transfer_model_5.fit(x=train_generator, 
                        validation_data=validation_generator,
                        epochs=500, 
                        steps_per_epoch=steps_per_epoch, 
                        validation_steps=validation_steps,
                        callbacks=[model_cp6, early_stopping, tensorboard])

        '''

        best_model = load_model('./CovNet_logs/Checkpoint2B.hdf5')

        metrics = score_model(best_model, test_generator, num_test)

        print(metrics)
