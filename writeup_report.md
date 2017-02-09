# **Behavioral Cloning** 

---

### **Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: images/network_model_behavioral_cloning.jpg "Model Visualization"
[image2]: images/center_normal.jpg "Grayscaling"
[image3]: images/left_recov.jpg "Recovery Image"
[image4]: images/center_recov.jpg "Recovery Image"
[image5]: images/right_recov.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image7]: images/data_plot_training.png "final_data_plot"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py:** containing the script to create and train the model
* **network_model.py:** containing the network. You can use the file to visualize the neural network.
* **drive.py:** for driving the car in autonomous mode
* **model.h5:** containing weights of a trained convolution neural network
* **model.json:** containing a trained convolution neural network
* **result___track___1.mp4:** I recorded the screen with with Xbox utility on windows for the track_1. Its inside video folder.
* **writeup_report.md:** summarizing the results
* **images:** containing images that are referenced in this file. 

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing.
Since, I have used json file to self drive the car please run following code to run:

```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

>My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 1164 (model.py lines 94-142) 

>The model includes **RELU** layers to introduce nonlinearity (every layer has it), and the data is normalized in the model using mehtod **process_image()** (line no 41). 

#### 2. Attempts to reduce overfitting in the model

>The model contains dropout layers in order to reduce overfitting (model.py lines 109,119,128). 

>The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-90). Also the images from the **data_generator()** were choosen randomly and I have also made use of left and right camera images to make the car learn better without overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

>The model used an **adam** optimizer, and I have set the learning rate as **0.0001** (model.py line 162).

#### 4. Appropriate training data

>Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to create the required dataset. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

> hit the sweet spot between how much data should I capture, what hyper-parameters should I choose, should there be any pre-processing of data and creating the overall neural network model.


My first step was to use a convolution neural network model similar to the ...

>what we have already used in the modules being taught like following:

>- con2D layer
- flatten layer
- dense layer
- output layer

>But soon it was evident how incable this model is. So, after reading a little more and skimming through slack channels and forums I got to know about Nvidia model which many was vouching to work for them. So, I started implementing that model first.

I thought this model might be appropriate because ...

>its from Nvidia and it has already worked for many in the class. But, it was not working for me in its exact form. Could be because of many reasons - more data requirement, changes to preprocessing method, etc. So given the data i was continuously capturing I started making changes to this base neural network model. I started experimenting with the number of neurons in each layer and also with different positioning for dropouts and adding/substracting layers to/from the base model while adding more and more training data. After every failure in the simualtor I was traing the car for recovery and standard driving conditions. For learning rate I started with 0.001 but later decide on 0.0001. I just wanted my model to learn slowly but correctly.

>Initially I had no idean what paramter to tune and how to set that up. I fixed the learning rate first. After that I exprimented with image size and batch size. Found out that with image size as 128 model was throwing out-of-memory error too often to test. Then I went to 32 and finally settle at 64 as it was giving me enough room to expriment with model and batch size. Then I fixed the batch size. Epochs I set to 5 as more than that was not producing any significant reduction in validaiton errors. From this point it was only the model I needed to tune.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

>I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was not learning. After adding more than 15000 images training loss was reducing. Validation loss was still more than 0.2-0.3 though. Only after 20000 images validation was reducing. As I kept exprimenting with the network model by adding more and more neurons in each layer, I saw that this was reducing the training loss but validation loss was not decreasing with respect to this and in some cases exceeding the previous values. It made be realize that my model has started overfitting.

To combat the overfitting, I modified the model so that ...

>there is a dropout after each layer. That made the condition worse as loss reduced a little bit but model was not learning as expected. Then I tried experimenting with strategically placing the dropout layer into the model instead of after every layer as I was doing earlier. With this experimenting I found that keeping a dropout every two layers were making the model learn better. There were still turns at the which the car would fall off but with each recovery session being added to data set movement of car was improving gradually.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

>ran the car 2-3 times on the same spot teaching it to recover and then run the model again. I repeated it several times for each time it fell off the track. Also, while teaching the car to recover I would make sure to follow the path as exactly as possible and make it recover so that more images are created for those scenarios to help the model learn.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from right yellow stripe and moving to center of road :

![alt text][image3]
![alt text][image4]
![alt text][image5]

>To augment the data set, I used left / right images and angle shiffting thinking that this would randomize the data more helping the learning. Also to increasing the number of non-zero steering angle I tried making very small changes to car motion while keep the car steady on road.

After the collection process, I had **2000-3000** number of data points. I then preprocessed this data by ...

>I read that using images from right and left camera did improve things a lot for many. So, with help of forum and slack I implemented a simple method to randomly choose from left/center/right image for every steering angle and make appropriate angle shiffting to counter its effect. This also helped a lot in randomizing the dataset and significantly increased the learning in my case. Apart from that I did usual normalization and random selection of images for each batch size with the help of data_generator().


I finally randomly shuffled the data set and put **20%** of the data into a validation set. 


>I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

>The ideal number of epochs was **5** as more than that was not adding significant reduction in validation loss. I used an **adam** optimizer but ended up setting the learning rate manually to 0.0001. It gave me more confidense over deciding changes in other parameters.