# Lier Detection using DeepLearning Technology

A lie detector based on Pytorch for a 2022 R&E school project.

![model_v1](https://github.com/yoonhero/lierhero/blob/master/docs/model_v1.png?raw=true)

## Introduction

![lie detector](https://img.fruugo.com/product/1/88/138417881_max.jpg)

I sometime play the lie game with my brother and sister. This equipment looks like real lie detector. But the fundamental law is simple. I'd like to improve this lie detector using Deep learning Technology.



## Presumption

I set up presumption that a lie relates with our face and our heart rate. Because when I lie, my heart beats fast and my facial expression is changed. 

I tracked face expression with Mediapipe FaceMesh model and heart rate with Arduino Sensor.


## Collecting Data 
 
I made the UI for data collecting using Flask webpage. This webpage is quite simple but essential for training. Using this UI, I get the image and heart rate numeric figure during lying and not lying.


## Model Training

<strong>MODEL V1</strong>

I made simple Linear Net with Pytorch. I will experiment many type of models to improve the performance. 

![model_v1](https://github.com/yoonhero/lierhero/blob/master/docs/model_v1.png?raw=true)




<strong>MODEL V2</strong>

I made Neural Network with Convolutional Network becauase linear net doesn't reflect the facial expresssion emotion. But this network doesn't working well too...


![model_v2](https://github.com/yoonhero/lierhero/blob/master/docs/model_v2.png?raw=true)

