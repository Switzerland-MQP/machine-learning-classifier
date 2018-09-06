from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 2 # This autoencoder will compress 784 floats down to 32

input_img = Input(shape=(784,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(2, activation='relu', name="bottleneck")(encoded)

decoded = Dense(32, activation='relu', name="decoder_input")(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded) #Map the input layer to its reconstruction


##############
#Now we create a model to get out the encoded vector
encoder = Model(input_img, encoded)
# Goes from the input image to the encoded vector



autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
####### The _ underscores are to discard the labels, which we don't need

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# dataset is Nx28x28, reshape to Nx784


# x_train is both input and output, therefore it learns how to 
# produce the same input and output but reduces dimension
autoencoder.fit(x_train, x_train,
								epochs=100,
								batch_size=256,
								shuffle=True,
								validation_data=(x_test, x_test))

# Encode images to length 32 vectors
encoded_imgs = encoder.predict(x_test)

# Decode images back to images
decoded_imgs = autoencoder.predict(x_test)

################### Results
import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	
	#Divide plot into 2 rows, and a column for each image
	#Put the original image on the top column
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	#ax.get_xaxis().set_visible(False)
	#ax.get_yaxis().set_visible(False)

	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	#ax.get_xaxis().set_visible(False)
	#ax.get_yaxis().set_visible(False)
	
plt.show()

# Try showing the encoded vectors as plots.
# Try reducing down to less than 32












