---

title: Intro to Neural Networks
subtitle: How I ran into DL?
summary: This is What my Thesis Project will be about.
authors:
- admin
date: "2016-04-27T00:00:00-03:00"
external_link: ""
math: true
image:
  caption: Stolen photo from [Google](https://www.google.com)
  focal_point: Center
links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/georgecushen
slides: example-slides


tags:
- Deep Learning


url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""
---


## What is a Artificial Neural Network?

The Picture you see up there is far a way from what a real Neural Network looks like:

Actually it is more similar to something like this:

<center>{{< figure library="1" src="ANN.jpg" title="Deep Leaning Network" numbered="true">}}</center>

The configuration showed above is nothing but a visual representation of serial Matrix Multiplications. The neural network (aka ANN that stands for Artificial Neural Networks) itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs^[Wikipedia: https://en.wikipedia.org/wiki/Artificial_neural_network].

ANN are inspired by Neurons but they do not emulate them. The main advantage of this system is that organizes succesive Matrix multiplications and provides a visual representation of the different steps in the Optimization algorithm.

There are different types of Networks, Densely Connected, Convolutional, Recurrent, Long Short Memory, Radial Biased, Autoencoders, and so on. All of them having their own area of specialization, strengths and shortcomings.

### Basic Structure

Every Network will contain Nodes/Units, emulating Neurons and Edges, emulating the connection between Units.

Normally the Units are organized by Layers, every Network should contain a first layer for Inputs, a last Layer for Outputs and Intermediate Layers, also known as Hidden Layers.

<center>{{< figure library="1" src="Basic_ANN.jpg" title="Simple Multi-Layer Perceptron" numbered="true">}}</center>

+ Input Nodes $ i_{j} $ will contain Input Values to train the Network. 

+ The Edges will provide weigths $ w_{j} $. 

+ Hidden Layer Nodes $ h_{j} $ will contain the result of Sum Product between Input Nodes and weighted edges connected to them. 

+ Finally Output Nodes  $ o_{j} $ will contain the result of Sum Product between hidden Nodes and the Edges Connected to them.

+ Optionally, Networks can include a Bias $ b_{j} $

The Network then will work in the following way: 










$$ h_{1} = w_{1} \cdot i_{1} + w_{2} \cdot i_{2} + b_{i} $$

= 0.15 \cdot 0.05 + 0.25 \cdot 0.10 + 0.35 = 0.3825 $$

$$ h_{2} = w_{2} i_{1} + w_{4} i_{2} + b_{i} = 0.20 \cdot 0.05 + 0.30 \cdot 0.10 + 0.35 = 0.39 $$

Operations in the NetworkOutput Values  will contain the result of the Network Calculation.

$$  h_{1} = w_{1} i_{1} + w_{2} i_{2} + b_{i} $$

$$ \sum_{n=1}^{10} n^2 $$
