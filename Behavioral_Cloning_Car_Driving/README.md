[image1]: ./images/nvidia_model.png "NVIDIA Model"

# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, I have used deep neural networks and convolutional neural networks to clone driving behavior. I have trained, validated and tested a model using Keras. The model output is a steering angle to an autonomous vehicle.

The project is having following five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* video.py (script to create video from images in a directory)
* model.h5 (a trained Keras model)
* writeup_report.md (a report writeup file)
* run1.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior or use Udacity provided dataset
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Details About Files In This Directory

### `model.py`
Usage of this is to train and validate the model so that vehicle is driving autonomously around the track. 
This also creates an h5 file i.e. `model.h5` to save the model. 

Following command can be used to executre this file -

```sh
python model.py
```

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

2019_02_20_13_46_24_755.jpg  2019_02_20_13_48_00_712.jpg  2019_02_20_13_49_08_200.jpg  2019_02_20_13_50_14_602.jpg
2019_02_20_13_46_24_837.jpg  2019_02_20_13_48_00_733.jpg  2019_02_20_13_49_08_225.jpg  2019_02_20_13_50_14_626.jpg
2019_02_20_13_46_24_867.jpg  2019_02_20_13_48_00_759.jpg  2019_02_20_13_49_08_307.jpg  2019_02_20_13_50_14_707.jpg
2019_02_20_13_46_24_894.jpg  2019_02_20_13_48_00_835.jpg  2019_02_20_13_49_08_331.jpg  2019_02_20_13_50_14_734.jpg
2019_02_20_13_46_24_979.jpg  2019_02_20_13_48_00_857.jpg  2019_02_20_13_49_08_354.jpg  2019_02_20_13_50_14_762.jpg
2019_02_20_13_46_25_010.jpg  2019_02_20_13_48_00_883.jpg  2019_02_20_13_49_08_437.jpg  2019_02_20_13_50_14_849.jpg
2019_02_20_13_46_25_043.jpg  2019_02_20_13_48_00_974.jpg  2019_02_20_13_49_08_463.jpg  2019_02_20_13_50_14_878.jpg
2019_02_20_13_46_25_127.jpg  2019_02_20_13_48_00_994.jpg  2019_02_20_13_49_08_486.jpg  2019_02_20_13_50_14_906.jpg
2019_02_20_13_46_25_156.jpg  2019_02_20_13_48_01_027.jpg  2019_02_20_13_49_08_563.jpg  2019_02_20_13_50_14_990.jpg
2019_02_20_13_46_25_182.jpg  2019_02_20_13_48_01_132.jpg  2019_02_20_13_49_08_590.jpg  2019_02_20_13_50_15_019.jpg
2019_02_20_13_46_25_265.jpg  2019_02_20_13_48_01_166.jpg  2019_02_20_13_49_08_614.jpg  2019_02_20_13_50_15_048.jpg
2019_02_20_13_46_25_292.jpg  2019_02_20_13_48_01_192.jpg  2019_02_20_13_49_08_693.jpg  2019_02_20_13_50_15_129.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

## Model Architecture

I have used NVIDIA model link [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

![alt text][image1]

We train the weights of our network to minimize the mean-squared error between the steering command output by the network, and either the command of the human driver or the adjusted steering command for off-center and rotated images. The network architecture shown above consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. 

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.

The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.

We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.



#### Model Hyperparameters
```
- Optimizer: Adam optimizer, so the learning rate was not tuned manually 
- Epoch: 5
- Batch size: 32
```

## Steps to follow to complete this project

* Get the data/images either through Udacity provided dataset or generate through Udacity provided Simulator.  
  
  `IMG folder` - this folder contains all the frames of your driving.
  
  `driving_log.csv` - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. We'll mainly be using the steering angle.
  
* Have created `append_data` function for Flipping Images And Steering Measurements

* Have created `images_and_measurements` function so that during training, when we can feed the left and right camera images to our model as if they were coming from the center camera. This way, we can teach your model how to steer if the car drifts off to the left or the right.

* Used NVIDIA model as shown in Model Architecture section

* Used Generators in `generator()` function 
    
    The images captured in the car simulator are much larger than the images encountered in the Traffic Sign Classifier Project, a size of 160 x 320 x 3 compared to 32 x 32 x 3. Storing 10,000 traffic sign images would take about 30 MB but storing 10,000 simulator images would take over 1.5 GB. That's a lot of memory! Not to mention that preprocessing data can change data types from an int to a float, which can increase the size of the data by a factor of 4.

    Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator we can pull pieces of the data and process them on the fly only when we need them, which is much more memory-efficient.
    
    A generator is like a coroutine, a process that can run separately from another main routine, which makes it a useful Python function. Instead of using return, the generator uses yield, which still returns the desired output values but saves the current values of all the generator's variables. When the generator is called a second time it re-starts right after the yield statement, with all its variables set to the same values as before.
    
* Used Keras method `fit_generator` to trains the model on data generated batch-by-batch by a Python generator. 
  The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.

* Save the Model to model.h5
* Execute the command `python model.py` to run the model.
* Change the speed in `drive.py` from default 9 to 12.
* Execute the command `python drive.py model.h5 run1` 
  The fourth argument, `run1`, is the directory in which to save the images seen by the agent
* Execute the command `python video.py run1` to create the video from images