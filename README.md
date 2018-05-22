# Semantic Segmentation

## Introduction
The goal of this project is to classify pixels of an image to find the drivable portion of the road using a Fully Convolutional neural network based on VGG16 architecture.

## Architecture
-   There are two classifications road and non-road.
-   Pretrained VGG-16 was converted to fully convolutional neural network using 1x1 convolutions instead of fully connected layers
-   Performance of network is improved by using 2 Skip connections between the 4th layer and the 1x1 convolutional layer of decoder (layer 7 that upsamples the image) and 3rd layer with 1x convolutional layer of the decoder(output layer that upsamples the image from previous function)
-  Used kernel size of 16X16 in the last layer

## Hyperparameters

-   Epoch: 20
-   Batch Size: 6
-   Keep Prob : 0.8
-   Learning Rate: 0.0001

## Results

-   The results of all the test images is in [runs](./runs/1526919296.7343135)  folder.
-   Training loss per batch after 1 epochs was around 0.6
-   Training loss per batch after 10 epochs was around 0.2
-   Training loss per batch at end of 20 epochs was around 0.15

## Sample outputs

[Img1](./output/um_000015.png)
[Img2](./output/um_000018.png)
[Img3](./output/um_000024.png)
[Img4](./output/um_000031.png)
[Img5](./output/um_000034.png)