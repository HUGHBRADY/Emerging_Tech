import numpy as np
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import gzip

# Open and read all of the MNIST files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:  
    raw_test_img = f.read()
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:  
    raw_test_lbl = f.read()
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:  
    raw_train_img = f.read()
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:  
    raw_train_lbl = f.read()

# Next we'll trimodelthe excess bytes at the start and convert the bytes to integers
test_img = np.frombuffer(raw_test_img, dtype = np.uint8, offset = 16) / 255
test_lbl = np.frombuffer(raw_test_lbl, dtype = np.uint8, offset = 8)
train_img = np.frombuffer(raw_train_img, dtype = np.uint8, offset = 16) / 255
train_lbl = np.frombuffer(raw_train_lbl, dtype = np.uint8, offset = 8)

# Finally we reshape the image arrays
test_img  =  test_img.reshape(10000, 784)
train_img = train_img.reshape(60000, 784)

num_classes = 10

# Convert class vectors to binary class matrices 
train_lbl = kr.utils.to_categorical(train_lbl, num_classes)
test_lbl = kr.utils.to_categorical(test_lbl, num_classes)

# Next we create our keras model
model = kr.models.Sequential()

# Layers to add to our sequential model. The first layer defines the input shape
model.add(Dense(1000, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))                                                 
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Defines the loss function, optimizer and metrics and is needed for training
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# fit trains the model for a given number of epochs
model.fit(train_img, train_lbl, batch_size=100, epochs=20, validation_data=(test_img, test_lbl))
score = model.evaluate(test_img, test_lbl, verbose=0)

# Finally, print the loss and the accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])