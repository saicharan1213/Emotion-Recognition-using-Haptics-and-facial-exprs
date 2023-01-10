from keras.models import Model,Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import  MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

''' convolution layer has been built using 32 filters each of 5x5 size and a stride of 1x1
activation function used was ReLu, the pooling layer choosen was max pooling, 3 hidden layers with 32, 64,128 neurons 
has been used respectively with ReLu activation and a output layer with 3 nurons to classify 3 classes with 
softmax activation function so as to generate the probability of each class
loss function - categorical crossentropy
optimizer - Adam with alpha as 1e-04'''

model = Sequential() #intializing sequential model
model.add(Conv2D(32, (5, 5), strides = (1, 1), name = 'conv0', input_shape = (48,48,1))) 
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(32, (2, 2), strides = (1,1), name="conv1"))
model.add(Conv2D(32, (2, 2), strides = (1,1), name="conv2"))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool1'))


model.add(Conv2D(64, (2, 2), strides = (1,1), name="conv3"))
model.add(Conv2D(64, (2, 2), strides = (1,1), name="conv4"))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool2'))
#dense layer
model.add(Flatten())
model.add(Dense(32, activation="relu", name='rl1'))
model.add(Dense(64, activation="relu", name='rl2'))
model.add(Dense(128, activation="relu", name='rl3'))
model.add(Dense(3,activation='softmax', name='sm'))


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-4), 
              metrics=['accuracy'])

model.summary()


train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.30)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)
batch_size = 16

#directory of train and validation should be given
train_generator = train_datagen.flow_from_directory(
    directory=r'archive\images\images\train',
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=45
)
valid_generator = train_datagen.flow_from_directory(
    directory=r'archive\images\images\validation',
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=45
)



history = model.fit(train_generator,validation_data = valid_generator,epochs = 70)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LOSS')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper right')
plt.show()