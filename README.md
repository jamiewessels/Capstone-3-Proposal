## A-CHORD-ing to the Image


#### Goal: 
The goal of this project is to create a piano chord classifier based on images.  

#### Background: 
Deep learning has been used to classify chords based on audio input, but there aren't many chord identifiers based on images. A chord image classifier could have many use cases such as helping hearing impaired musicians, supporting virtual learning, video transcription, and cuing theatrical changes for large productions. There is also an opportunity to build an ensemble model, combining the audio and image classifiers. 

#### Collecting the Data
With a "little help from my friends", I generated and self-labeled images of three chords: C, F, and G.  The images came from 11 ppl and 8 unique pianos, included both left and right hands and multiple fingerings.  There were approximately XX images generated per person. 

<p align="center">
<img src="images/brady_bunch_hands.png" width="600px" >
</p>

I randomly split the raw images into training, validation, and test sets, and then performed augmentation on the training set using the kera's ImageDataGenerator.  This created 20 additional images per raw training image, by randomly performing transofmrations such as rotation, shearing, zoom, shifts.

### Fitting a Model
The model was built with the Xception architecture and was initialized to the ImageNet weights.   

[Here talk about process!! and fix the graph below

maybe a flow chart of what improvements made over time ]


<p align="center">
<img src="images/model_progress.jpeg" width="600px" >
</p>


### Technology Used


The data set is biased in the following ways:
1. Small number of individuals and pianos 
1. 9 out of 11 individuals are white
1. All images taken during the day

