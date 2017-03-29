import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.applications import VGG16

#constants
training_samples = 2000
validation_samples = 800
batch_size = 16
epochs = 50
train_data_dir = "data/train/"
validation_data_dir = "data/validation"
weight_path = "bottleneck_features_model.h5"

#Data Generators
train_data_gen = ImageDataGenerator(rescale=1./255, 
                                    shear_range=30,
                                    zoom_range=0.5,
                                    height_shift_range=0.2,
                                    width_shift_range=0.2)

validation_data_gen = ImageDataGenerator(rescale=1./255)

train_data = train_data_gen.flow_from_directory(train_data_dir,
                                                target_size=(150, 150),
                                                class_mode="binary",
                                                batch_size=16)
validation_data = validation_data_gen.flow_from_directory(validation_data_dir,
                                                          target_size=(150, 150),
                                                          class_mode="binary",
                                                          batch_size=16)

#pretrained model
input_layer = Input(shape=(150, 150, 3))
base_model = VGG16(include_top=False, weights="imagenet", input_tensor=input_layer)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(64, activation="relu"))
top_model.add(Dense(1, activation="sigmoid"))
#load weights
top_model.load_weights(weight_path)

#custom model on top of VGG1
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

#untrained layers
for layer in model.layers[:15]:
  layer.trainable = False

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy", "binary_crossentropy"])

history = model.fit_generator(
                     train_data, 
                     steps_per_epoch=training_samples // batch_size,
                     epochs=epochs,
                     validation_data=validation_data,
                     validation_steps=validation_samples // batch_size)




















