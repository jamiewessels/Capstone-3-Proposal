## A-CHORD-ing to the Image


#### Goal: 
The goal of this project is to create a piano chord classifier based on images.  

#### Background: 
Deep learning has been used to classify chords based on audio input, but there aren't many chord identifiers based on images! A chord image classifier could have many use cases such as helping hearing impaired musicians or supporting virtual learning, and could be a precursor to musical transcription based on a video input. 

#### Collecting the Data
With a "little help from my friends", I generated and self-labeled images of five chords: C, F, Am, Dm, and G.  The images came from 12 people and 8 unique pianos, included both left and right hands and multiple fingerings.  There were approximately XX images generated per person. 

<p align="center">
<img src="images/brady_bunch_hands.png" width="600px" >
</p>

I randomly split the raw images into training and validation sets, and then performed augmentation on the training set using the kera's ImageDataGenerator.  This created 20 additional images per raw training image, by randomly performing the following transformations: rotation, shearing, dilation, and translations.

### About the Model
I chose to use the Xception architecture, initialized to the ImageNet weights (add link to paper, keras.applications). This allowed me to leverage prior learning to accommodate a smaller sample size. The only change made to the original Xception architecture was the output layer, which was set to a 5-neuron softmax output. 

The figure below shows a baic Xception structure.  Colors indicate that the weights were adjusted from the original ImageNet weights. The final model and associated weights is saved as best_model_5chords.hdf5 within the CovNet_logs folder. 

Xception Model Structure:

<p align="center">
<img src="images/Xception_model.png" width="600px" >
</p>


### Tuning Process
Layer weights were sequentially unfrozen and the learning rate and/or optimizer was during the training process. Early stopping was used to monitor validation loss during each training iteration. The loss function was set to sparse categorical cross-entropy. The model started as a 3-chord classifier, but was later changed to a 5-chord classifier after promising results.  When switching to the 5-chord classifier, the weights from the best 3-chord classifier were used. 

First Tuning Process: 3 Chord Classifier
1. Outer layer - 3 class, softmax: SGD(lr = 0.2), 10 epochs
1. Block 14: SGD(lr=0.01, momentum = 0.9, decay = 0.001)
1. Block 13: SGD(lr=0.01, momentum = 0.9, decay = 0.001)
1. Block 12: SGD(lr=0.001, momentum = 0.9, decay = 0.001)
1. Block 12: Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 11: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 10: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)
*changed to a 5-chord model*
1. Outer layer - 5 class, softmax: SGD(lr = 0.2), 10 epochs 
1. Block 14: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 13: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 12: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 10+: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)

I chose to monitor validation accuracy, as I cared most about my model correctly classifying the chords. The graph below outlines the validation accuracy with each training iteration. We see the biggest jump in accuracy during the 3rd iteration, after unfreezing block 13 (layers 116+). The model ended with a validation accuracy of 98.3%. 

<p align="center">
<img src="images/model_progress.jpeg" width="600px" >
</p>

### Model Performance
The figure below shows a normalized confusion matrix for the 5 chord classes. Of the test images, the model was able to correctly label all true F and G chords.  The model had the lowest recall for chord Dm (95%); the model  incorrectly labeled 5% of the true Dm chords as Am.    it incorrectly labeled the chord Am 5% of the time.  


<p align="center">
<img src="images/conf_matrix_5chords_chkpt4.png" width="600px" >
</p>


### Model in Action

**Clip of me!**
The model correctly labels each chord.  The model's prediction fluctuates when keys are not being pressed. 
<p align="center">
<img src="images/videos/video_predictions/4x_self_1.gif", width="400px" >
</p

**Another clip of me!**
The model incorrectly labels the Dm chord (2nd in the loop) as a G. Dm had the lowest recall in the model.

<p align="center">
<img src="images/videos/video_predictions/4x_self_2.gif", width="400px" >
</p


**Clip of youtuber!**
The model performs well on this youtube clip.  
*Things to note:* The clip was cropped because the backdrop also looked like piano keys.  
<p align="center">
<img src="images/videos/video_predictions/4x_youtube.gif", width="400px" >
</p

### Predictions

### Model Limitations
