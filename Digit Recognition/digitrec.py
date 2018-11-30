import numpy as np
import keras
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

# Next we'll trim the excess bytes at the start and convert the bytes to integers
test_img = np.frombuffer(raw_test_img, dtype = np.uint8, offset = 16)
test_lbl = np.frombuffer(raw_test_lbl, dtype = np.uint8, offset = 8)
train_img = np.frombuffer(raw_train_img, dtype = np.uint8, offset = 16)
train_lbl = np.frombuffer(raw_train_lbl, dtype = np.uint8, offset = 8)

# Finally we reshape the image arrays
test_img = test_img.reshape(10000, 28, 28)
train_img = train_img.reshape(60000, 28, 28)
