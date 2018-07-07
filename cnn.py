from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

#initalizing CNN
classifier=Sequential()

#defining convolution layer one
classifier.add(Convolution2D(6,(7,7),
                input_shape=(48,48,3),
		padding='same',
                activation='relu'))

#pooling layer one
classifier.add(MaxPooling2D(pool_size=(2,2)))

#defining convolution layer two
classifier.add(Convolution2D(6,(5,5),
                input_shape=(24,24,3),
		padding='same',
                activation='relu'))

#pooling layer two
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim=256,activation='relu'))
classifier.add(Dense(output_dim=6,activation='softmax'))

#compile
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

#Image preprocessing and fitting to CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 featurewise_center=True)

test_datagen=ImageDataGenerator(1./255)

training_set=train_datagen.flow_from_directory('./dataset/train_set',
                                               target_size=(48,48),
                                               batch_size=60,
                                               class_mode='categorical')

testing_set=test_datagen.flow_from_directory('./dataset/test_set',
                                               target_size=(48,48),
                                               batch_size=60,
                                               class_mode='categorical')

#checkpointing
checkpoint=ModelCheckpoint('best_epoch.h5', monitor='val_loss',  verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callbacks_list=[checkpoint]

classifier.fit_generator(training_set,
                         samples_per_epoch=12600,
                         nb_epoch=25,
                         validation_data=testing_set,
                         nb_val_samples=1500,
			 callbacks=callbacks_list)

