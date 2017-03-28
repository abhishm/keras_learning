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
weights_path = ""

#Data Generators


#pretrained model
input_layer = Input(shape=(150, 150, 3))
base_model = VGG16(include_top=False, weights="imagenet", input_tensor=input_layer)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(64, activation="relu"))
top_model.add(Dense(1, activation="sigmoid"))
#load weights
top_model.load_weights("bottleneck_features_model.h5")

#custom model on top of VGG1
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))




















