# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/training_dataset_visualization.png "Visualization"
[image2]: ./writeup_images/original_augmented_image.png "Original and Augmented Image"
[image3]: ./writeup_images/hist_equalization.png "Histogram Equalization"
[image4]: ./writeup_images/gray_scale.png "Gray Scale"
[image5]: ./writeup_images/normalization.png "Normalized Images"
[image6]: ./web_images/4_speed_limit_70.jpg "Traffic Sign 1"
[image7]: ./web_images/9_No_passing.jpg "Traffic Sign 2"
[image8]: ./web_images/14_stop.jpg "Traffic Sign 3"
[image9]: ./web_images/17_No_Entry.jpg "Traffic Sign 4"
[image10]: ./web_images/22_bumpy_road.jpg "Traffic Sign 5"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. A basic summary of the data set.

This has been provided using basic python and code can be found in the Step 1 under Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training dataset. It is a bar chart showing how the training data is distributed among classes. We can clearly see that dataset is skewed.

As can be seen in the Step 1 of notebook, the data distribution is almost similar in train, valid and test datasets. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing steps:

* First training dataset was augmented to get more data, in such a way that each class have around 5000 images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

* Histogram equalization was applied i.e. the pixel intensity of each image is averaged over all three channels.

We plot the histogram of all images in the training set. The pixel intensity of each image is averaged over all three channels and a histogram is then calculated over all images in the set. 

The histogram has a large hump between intensity levels 0 and 50 and a tall spike around intensity level 255. This suggests a large number of very-dark pixels or very bright pixels.

To get a clearer picture of the overall brightness, a cumulative histogram is plotted further below. As described here (https://luminouslandscape.com/understanding-histograms/), if the intensity levels are divided into 5 bins, very dark, dark, medium, light and very light, close to 60% of pixels fall in the very dark, dark categories. 

Compared to this only about 10% of pixels fall in the light and very light regions. This suggests widespread underexposure and histogram equalization may improve training performance.

Here is an example of an image after histogram equalization:

![alt text][image3]

* Converted the images to gray scale

![alt text][image4]

* Apply Normalization to Images

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    |  1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| 800,   outputs 512							                        |
| RELU					|												|
| Fully connected		| 512,   outputs 256 			                        			|
| RELU					|												|
| Softmax				| 256,    outputs 43      					                        		|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following hyperparameters:
 * Epochs - 30
 * batch size - 128
 * learning rate - 0.001
 * optimizer - Adam Optimizer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 97.9%
* test set accuracy of 95.7%

I choose an iterative approach -
* First approach
    * train dataset - original images
    * Model architecture - 
       * conv1 : 32x32x3 to 28x28x6
       * activation: relu
       * pooling layer : 28x28x6 14x14x6
       * conv2 : 14x14x6 to 10x10x16
       * activation: relu
       * pooling layer : 10x10x16 to 5x5x16
       * fc1 layer - 400 to 120
       * activation : relu
       * fc1 layer - 120 to 84
       * activation : relu
       * softmax layer - 84 to 43
    * hyperparameters - 
       * epochs - 30
       * batch size - 128
       * learning rate - 0.0005 
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 90.8
    * Result - overfitting

* Second approach
    * train dataset - original images
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 30
       * batch size - 128
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 99.6
       * validation - 90.4
    * Result - Accuracy reduced in fraction but still overfitting  

* Third approach
    * train dataset - original images
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 30
       * batch size - 64
       * learning rate - 0.0005
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 92.1
    * Result - Accuracy increased with decreased batch size
    
* Fourth approach
    * train dataset - original images
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 30
       * batch size - 64
       * learning rate - 0.0005
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 87.6
    * Result - Accuracy decreased a lot with change in lerning rate

* Fifth approach
    * train dataset - Normalized data
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 30
       * batch size - 128
       * learning rate - 0.0005
       * optimizer - Adam
    * Accuracy - 
       * training - 100
       * validation - 89.4
    * Result - Model is overfitting

* Sixth approach
    * train dataset - Normalized data
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 50
       * batch size - 128
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 100
       * validation - 93.3
    * Result - A good improvement in the accuracy with these hyperparameters

* Seventh approach
    * train dataset - Normalized data
    * Model architecture - Same as in first approach
    * hyperparameters - 
       * epochs - 50
       * batch size - 64
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 100
       * validation - 94.4
    * Result - Again an improvement in the accuracy with these hyperparameters. It actually crossed the requirements for Rubric to reach 93% validation accuracy.
   
* Eighth approach
    * train dataset - Gray scale and Normalized data
    * Model architecture
       * conv1 : 32x32x1 to 28x28x6
       * activation: relu
       * pooling layer : 28x28x6 14x14x6
       * conv2 : 14x14x6 to 10x10x16
       * activation: relu
       * pooling layer : 10x10x16 to 5x5x16
       * fc1 layer - 400 to 120
       * activation : relu
       * fc1 layer - 120 to 84
       * activation : relu
       * softmax layer - 84 to 43
    * hyperparameters - 
       * epochs - 50
       * batch size - 64
       * learning rate - 0.0005
       * optimizer - Adam
    * Accuracy - 
       * training - 100
       * validation - 94.3
    * Result - Almost same accuracy as in seventh approach
 
* Ninth approach
    * train dataset - Gray scale and Normalized data
    * Model architecture - Same as in eighth approach
    * hyperparameters - 
       * epochs - 50
       * batch size - 64
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 95.3
    * Result - An improvement in the accuracy as per last approach
 
* Tenth approach
    * train dataset - Histogram equalization, Gray scale and Normalization
    * Model architecture -
       * conv1 : 32x32x1 to 28x28x16
       * activation: relu
       * pooling layer : 28x28x16 14x14x16
       * conv2 : 14x14x16 to 10x10x32
       * activation: relu
       * pooling layer : 10x10x32 to 5x5x32
       * fc1 layer - 800 to 512
       * activation : relu
       * fc1 layer - 512 to 256
       * activation : relu
       * softmax layer - 256 to 43  
    * hyperparameters - 
       * epochs - 50
       * batch size - 128
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 87.2
    * Result - A sharp dip in accuracy, looks like model need to be deeper
 
 * Eleventh approach
    * train dataset - Histogram equalization, Gray scale and Normalization
    * Model architecture - Same as in thenth approch
    * hyperparameters - 
       * epochs - 50
       * batch size - 128
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 99.6
       * validation - 97.9
       * test - 94.9
    * Result - A good accuracy to stop experiments 
 
 * Twelvth approach
    * train dataset - Data Augmentation, Histogram equalization, Gray scale and Normalization
    * Model architecture - Same as in thenth approch
    * hyperparameters - 
       * epochs - 50
       * batch size - 128
       * learning rate - 0.001
       * optimizer - Adam
    * Accuracy - 
       * training - 99.9
       * validation - 96.2
       * test - 95.7
    * Result - An improvement but not much significant even after augmenting the dataset
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

* The first image(Speed Limit 70) should be easy to identify because it is pretty clean. 
* Secoond image (No Passing) might be difficult because there is a reflection and tree in the image. 
* Third image (No entry) should be esy to identify because this image is also clean.
* Fourth image (No Entry), I cannot be sure about this because there will be lot of noise after doing pre-processing steps.
* Fifth image (bumpy road) must be identify correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 70      		| Speed Limit 70  									| 
| No Passing     			| Yield 										|
| Stop					| Spped Limit 60											|
| No Entry	      		| Speed Limit 100					 				|
| Bumpy Road			| Bumpy Road      							|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This accuracy is not even half the test accuracy. I believe image size to be the main factor because to fit the model it need to bve 32x32x1.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 71th cell of the Ipython notebook.

For the first image, the model is perfctly sure that this is a speed limit 70 sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| speed limit 70   									| 
| .00     				| Pedestrians 										|
| .00					| speed limit 20 										|
| .00	      			| speed limit 30 				 				|
| .00				    | speed limit 50     							|


For the second image, the model is relatively sure that this is a yield sign (probability of 0.6), and the image does not contain a yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Yield  									| 
| .37      				| speed limit 60										|
| .01					| Bicycles Crossing								|
| .01	      			| End of all Speed and passing limits		 				|
| .01				    | Priority Road      							|

For the third image, the model is relatively sure that this is a speed limit 60 sign (probability of 0.69), and the image does not contain a speed limit 60 sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .69         			| speed limit 60   									| 
| .31     				| speed limit 100 										|
| .00					| speed limit 50											|
| .00	      			| speed limit 80		 				|
| .00				    | speed limit 20   							|

For the fourth image, the model is relatively sure that this is a speed limit 100 sign (probability of 0.96), and the image does not contain a speed limit 60 sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| speed limit 60  									| 
| .03     				| speed limit 60 										|
| .01					| speed limit 60											|
| .00	      			| No Passing				 				|
| .00				    | No Vehicles      							|

For the fifth image, the model is absolutely sure that this is a bumpy road sign (probability of 1.0), and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy Road  									| 
| .00     				| Bicycles crossing									|
| .00					| Road work									|
| .00	      			| Beware of ice/snow			 				|
| .00				    | No passing  						|


