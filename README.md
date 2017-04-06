In this reposiory, I trained a small image dataset using Keras high level api. I used three approaches:

- Create a Convolutional Neural Network from scratch and train it
- Use the VGG16 to get the features of the images and train a fully-connected neural network on it
- Combine VGG16 and the trained MLP in the previous example and fine-tune the results

### The Data Set
The data set contains images of dogs and cats. The goal is to create a classifier that can distinguish between a cat image and a dog image. The training dataset containes 8000 images. 4000 images were cat and 4000 images were dog. The validation dataset consist of 800 images -- 400 dog images and 400 cat images. 

#### The Data Augmentation
I used the keras image-processing library for data augmentation. The keras image processing library can do different random transformations to an image such as horizontal flip, shear, zoom, horizontal and vertical translation are to name a few. Consequently, it can make the Conv-net robust to the real transformation that we see in the real world. Moreover, it makes the mini batch-gradient descent method unbiased by always sampling a new dataset from training. 
 
### Convolutional Neural Network from Scratch

- I trained a small conv-net using the data. 
- The images are cropped to size 150 * 150. 
- The conv-net has three hidden layers. The first layer has 32 filters and each of the size 3x3. The first layer is followed by a polling layer. The second layer has 32 filters each with size 3x3. A pooling layer followed the second layer. The third layer has 64 filters with size 3x3. A pooling layer followed the third layer. After that, I put a fully connected layer with $64$ hidden neurons. Subsequently, a fully-connected layer with one output neuron follows it. All the activation expect the output layer is ReLu. The activation of the output layer is Sigmoid. 
- Loss: the loss is binary cross entropy loss.
- **Results**     
Loss:
>>> 
![loss_image](images/loss-epoch.png "loss vs epoch for conv-net")

Accuracy:
>>>
![accuracy_image](images/accuracy-epoch.png "accuracy vs epoch for conv-net")

#### Training Speed:
My model was processing 400 images / second. One training epoch took ~20 second on GPU and it took 130 seconds on CPU. 

#### Conclusion:
1. The training and validation accuracy is almost same which implies that the model is underfitting. This is not surprising because we are using a small conv-net for fitting the data. 
