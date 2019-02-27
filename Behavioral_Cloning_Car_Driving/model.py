import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Cropping2D, Lambda, Dropout
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from keras.layers.pooling import MaxPooling2D


def append_data(col, images, measurement, steering_measurements):
    current_path = image_path + '/' + col.strip()
    
    # read the image
    image = cv2.imread(current_path)
    # convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(np.asarray(image))
    steering_measurements.append(measurement)
    
    # random flipping
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # Flipping Images And Steering Measurements
        image_flipped = np.fliplr(image)
        images.append(np.asarray(image_flipped))
        measurement_flipped = measurement * (-1)
        steering_measurements.append(measurement)
          
# images_and_measurements functions read in images from center, left and right cameras. 
# It then create adjusted steering measurements for the side camera images then 
# returns images and angles to dataset x_total and y_total
def images_and_measurements(sample):
    images = []
    steering_measurements = []
    for line in sample[0:]:
        measurement = float(line[3])
        ## random data
        camera = np.random.choice(['center', 'left', 'right'])
        if camera == 'center':
            col_center = line[0]
            append_data(col_center, images, measurement, steering_measurements)
        elif camera == 'left':
            col_left = line[1]
            append_data(col_left, images, measurement + 0.25, steering_measurements)
        else:
            col_right = line[2]
            append_data(col_right, images, measurement - 0.25, steering_measurements)
    return images, steering_measurements

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]           
            images = []
            measurements = []
            for image, measurement in batch_samples:
                images.append(image)   
                measurements.append(measurement)
            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(x_train, y_train)

# Here starts the main funtions
if __name__ == '__main__':

    '''Read data'''
    # row in log path is IMG/<name>
    image_path = 'data'
    # read the driving_log csv file and store the records into a list `rows`
    # this csv file contains the file path for images for the left, center and right cameras
    #as well as information about the steering measurement, throttle, brake and speed of the vehicle.
    driving_log_path = 'data/driving_log.csv' 
    rows = []
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    # images_and_measurements functions read in images from center, left and right cameras. 
    # It then create adjusted steering measurements for the side camera images then 
    # addup images and angles to dataset x_total and y_total
    X_total, y_total = images_and_measurements(rows[1:])
    
    #build the NVIDIA model
    model = Sequential()
    #cropping images
    model.add(Cropping2D(cropping = ((74,20), (60,60)),input_shape=(160, 320, 3)))
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))  
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    '''Training: using MSE for regression'''
    model.compile(loss='mse', optimizer='adam')


    print('Training model')            
    samples = list(zip(X_total, y_total))      
    # split the dataset into 80% training dataset and 20% validation dataset
    train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
    # train the model using the generator function
    train_generator = generator(train_samples, batch_size = 32)
    # validate the model using the generator function
    validation_generator = generator(validation_samples, batch_size = 32)
    
    #history object contains the training and validation loss for each epoch    
    history_object = model.fit_generator(train_generator,
                                        samples_per_epoch = len(train_samples),
                                        validation_data = validation_generator,
                                        nb_val_samples = len(validation_samples),
                                        nb_epoch = 5, 
                                        verbose = 1)
    print('Endding training, starting to save model')
    # save the model
    model.save('model.h5')