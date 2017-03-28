import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.applications import VGG16

#Constants
training_samples = 2000
validation_samples = 800
batch_size = 16
epochs = 50
train_data_dir = "data/train/"
validation_data_dir = "data/validation/"

# pretrained model
model = VGG16(include_top=False, weights="imagenet")

def save_bottleneck_features():
  # Data Generators
  datagen = ImageDataGenerator(rescale = 1./255)
  
  train_data_generator = datagen.flow_from_directory(
                                     train_data_dir,
                                     target_size=(150, 150),
                                     batch_size=batch_size,
                                     class_mode=None, 
                                     shuffle=False)

  train_data = model.predict_generator(train_data_generator, 
                                       steps=training_samples // batch_size)

  validation_data_generator = datagen.flow_from_directory(
                                     validation_data_dir,
                                     target_size=(150, 150),
                                     batch_size=batch_size,
                                     class_mode=None, 
                                     shuffle=False)

  validation_data = model.predict_generator(validation_data_generator, 
                                            steps=validation_samples // batch_size)

  ## Save the data
  np.save(open("bottleneck_train_features.npy", "wb"), train_data)
  np.save(open("bottleneck_validation_features.npy", "wb"), validation_data)


#save_bottleneck_features()
def build_and_train_model():
  model = Sequential()
  model.add(Flatten(input_shape=X_train.shape[1:]))
  model.add(Dense(64, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(
        optimizer="rmsprop", 
        loss="binary_crossentropy",
        metrics=["accuracy"])

  history = model.fit(
                  X_train, y_train, 
                  validation_data=(X_validation, y_validation), 
                  batch_size=batch_size, epochs=epochs)

  return model, history

#save_bottleneck_features()

X_train = np.load("bottleneck_train_features.npy")
y_train = np.array([0] * 1000 + [1] * 1000)

X_validation = np.load("bottleneck_validation_features.npy")
y_validation = np.array([0] * 400 + [1] * 400)

top_model, history = build_and_train_model()
top_model.save("bottleneck_features_model.h5")


