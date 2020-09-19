## A-CHORD-ing to the Image


#### Goal: 
The goal of this project was to create a piano chord classifier based on images of hands playing the piano.  

#### Background: 
Deep learning has been used to classify chords based on audio input, but there aren't many chord identifiers based on images! A chord image classifier could have many use cases such as helping hearing impaired musicians or supporting virtual learning, and could be a precursor to musical transcription based on a video input. 

#### Collecting the Data
With a "little help from my friends", I generated and self-labeled images of five chords: C, F, Am, Dm, and G.  The images came from 12 people and 8 unique pianos, included both left and right hands and multiple fingerings.  There were approximately 15 images generated per person, per class. An assortment of the images is shown below.

<p align="center">
<img src="images/brady_bunch_hands.png" width="600px" >
</p>

I randomly split the raw images into training and validation sets, and then performed augmentation on the training set using keras' ImageDataGenerator.  This created 20 additional images per raw training image, by randomly performing the following transformations: rotation, shearing, dilation, and translations.

### About the Model
I chose to use the Xception architecture, initialized to the ImageNet weights (add link to paper, keras.applications). This allowed me to leverage prior learning to accommodate a smaller sample size. The only change made to the original Xception architecture was the output layer, which was set to a 5-neuron softmax output. 

The figure below outlines Xception's structure.  Colors indicate that the weights were adjusted from the original ImageNet weights. The final model and associated weights is saved as best_model_5chords.hdf5 within the CovNet_logs folder. 

**Xception Model Structure:**

<p align="center">
<img src="images/Xception_model.png" width="600px" >
</p>


### Tuning Process
During the training process, layers were sequentially unfrozen and the learning rate and/or optimizer was adjusted. Early stopping was used to monitor validation loss during each training iteration, and the loss function was set to sparse categorical cross-entropy. The model started as a 3-chord classifier, but was later changed to a 5-chord classifier after promising results.  When switching to the 5-chord classifier, the weights from the best 3-chord classifier were used. 

*initial tuning: 3-chord classifier*
1. Outer layer - 3 class, softmax: SGD(lr = 0.2), 10 epochs
1. Block 14: SGD(lr=0.01, momentum = 0.9, decay = 0.001)
1. Block 13: SGD(lr=0.01, momentum = 0.9, decay = 0.001)
1. Block 12: SGD(lr=0.001, momentum = 0.9, decay = 0.001)
1. Block 12: Adam(lr=0.001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 11: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 10: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)

*changed to a 5-chord classifier*

1. Outer layer - 5 class, softmax: SGD(lr = 0.2), 10 epochs 
1. Block 14: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 13: Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 12: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)
1. Block 10+: Adam(lr=0.00001, beta_1 = 0.9, beta_2 = 0.999)

I chose to monitor validation accuracy, as I cared most about my model being able to correctly classify the chords. The graph below outlines the validation accuracy with each training iteration. We see the biggest jump in accuracy during the 3rd iteration, after unfreezing block 13 (layers 116+). The model ended with a validation accuracy of 98.3%. 

<p align="center">
<img src="images/model_progress.jpeg" width="600px" >
</p>

### Model Performance
The figure below shows a normalized confusion matrix for the 5 chord classes. Of the test images, the model was able to correctly label all true F and G chords.  The model had the lowest recall for chord Dm (at 95%); the model incorrectly labeled 5% of the true Dm chords as Am.


<p align="center">
<img src="images/conf_matrix_5chords_chkpt4.png" width="600px" >
</p>


### Model in Action
I used OpenCV to predict chords based on video frames. The three gifs below show these prediction outputs (at 4x the original speed), and the original output videos can be found in the images/videos/predictions folder. It should be noted that there is still a lot more work to be done with the video classifier, specifically with pre-processing; for certain videos there seemed to be a discrepancy between the input color channels and the received color channels, which affected my model's capability.  To ensure that a video was compatible with the model, I ran still frames through my predict.py file and compared them to the video predictor's output.  

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
One concern I had was that my model had only seen a small number of hands and pianos, and of those hands, the majority were white. Therefore, it was important to me that I could test my model on additional images I could find online. This task was actually very difficult; surprisingly, many pictures of piano chords don't actually have hands in them! Additionally, the ones that do have hands in them are not always playing a triad (1-3-5 chords), like I was identifying in my model. The pictures below show un-seen hands, pulled from google and/or provided additional friends. Three of the images were incorrectly classified, indicating that there is more work to be done to diversify the training data!  


<p align="center">
<img src="images/additional_predicts.png" width="600px" >
</p>

### Moving Forward
It is clear that we need a more robust dataset! However, these weights are a good starting point for a chord classifier. 

**Multi-label Classification:** I think that a multi-label classifier that identifies which keys are being pressed would be able to compl.  This would provide more flexibility in what chords could be identified.


Diversify the dataset! 
More chords
Multi label classification - what keys are being depressed
