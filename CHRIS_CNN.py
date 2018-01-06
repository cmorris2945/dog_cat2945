#CONVOLUTIONAL NEURAL NETWORK PROJECT

#Installing THEANO 
#PIP install --upgrade --no-deps git+git://github.com/Theano.git

#Installing Tensorflow
#Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started

#Istalling Keras
#PIP install --upgrade keras

# part 1 - This is where I am BUilding my convolutional neural network HERE
#( NO need to preprocess the data becasue we already did it jackass)
#importing the KERAS libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  #This is where you add the "Fully connected" layer to learn the convolutional stuff

#intializing the CNN
classifier = Sequential()       #classifies the image to tell if it is a dog or a cat

# STEP 1 Convolution layer
classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3), activation= 'relu'))  #the activation function are like neurons and how much they can pass on the signal

#STEP 2 POOLING LAYER
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# THIS IS WHERE I AM ADDING A SECOND CONVOLUTIONAL LAYER
classifier.add(Convolution2D(32, 3, 3, activation= 'relu')) # when applying a second convolutional layer you don't need the "input_shape" function
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#STEP 3 FLATTENING LAYER
classifier.add(Flatten())  #No need to put arguments in the "flatten" function

# STEP 4 THE FULLY CONNECTED LAYER, OR THE LAST PART OF THE CNN CALLED THE ANN
classifier.add(Dense(output_dim = 128, activation = 'relu' ))     #the output_dim is the numebr of nodes in the hidden layer. Right here I am putting how many hidden nodes.
classifier.add(Dense(output_dim = 1, activation = 'sigmoid' ))   

# Now we COMPILE the CNN
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])        #you would use "categorical cross entropy" if you had more than TWO things you are comparing. But becuase there are only TWO: dogs and cats, we use "binary_cross_entropy" thing here       

# PART 2 FITTING THE CNN TO THE IMAGES
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #creates an object to preprocess the images of the test set

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(64, 64), #make sure you put the same parameters in the "convolutional input_shape thing: 64 and 64 ***NOTE***: One way to increase accuracy when running the CNN is the put higher number in the  "target size". Because you get more information from the pixels from the rows and columns that are added
                                               batch_size=32,
                                               class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/single_prediction',
                                            target_size=(64, 64),#use convolutioan layer input here too  ***NOTE***: One way to increase accuracy when running the CNN is the put higher number in the  "target size". Because you get more information from the pixels from the rows and columns that are added
                                            batch_size=32,
                                            class_mode='binary')  #binary outcome. Stays the same

classifier.fit_generator(training_set,
                        samples_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,  #put the test set here wehre you want to evaluate the performance
                        nb_val_samples = 2000)

# PART 3 MAKING NEW PREDICTIONS BASED OFF THE SINGLE DATASET OR THE DOG AND CAT PICTURE

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64, 64) )
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis =0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction ='dog'
else:
    prediction = 'cat'



