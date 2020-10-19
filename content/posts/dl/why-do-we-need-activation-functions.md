---
title: "Why Do We Need Activation Functions?"
date: 2020-09-09T00:42:27+05:30
draft: false
#featuredImage: "huggingface.png"
featuredImagePreview: "/posts/dl/images/activation.png"
coverImage: "/posts/dl/images/activation.png"
images: ["/posts/dl/images/activation.png"]
tags: ["Deep Learning", "Activation Functions"]
#categories: ["Deep Learning"]
description: "By now, we all are familiar with neural networks and its architecture (input layer, hidden layer, output layer) but one thing that I’m continuously asked is - ‘why do we need activation functions?’ or ‘what will happen if we pass the output to the next layer without an activation function’ or ‘Is nonlinearities really needed by the neural networks?’"

---
<!--more-->

{{< figure src="/posts/dl/images/activation.png" >}}

By now, we are familiar with neural networks and its architecture (input layer, hidden layer, output layer) but one thing that I’m continuously asked is - *"why do we need activation functions?"* or *"what will happen if we pass the output to the next layer without an activation function"* or *"Is nonlinearities really needed by the neural networks?"*

To answer the above questions, let us take a step back and understand what happens inside a neuron:


Inside a neuron, each input gets multiplied with the weights  $$(x * w) $$ Then, they are summed up $$∑(x * w)$$ Then, a bias is added $$∑(x * w) + b$$ And then, This output is passed to an activation function. Mathematically, $$y = σ (∑(x * w) + b)$$ where $σ$ is any activation function.


<br><br>

> An activation function simply defines when a neuron fires. Consider it a sort of tipping point: Input of a certain value won’t cause the neuron to fire because it’s not enough, but just a little more input can cause the neuron to fire. 

In the real-world data, as we model
observations using multiple features, each of which could have a varied and
disproportional contribution towards determining our output classes. 

In fact, our world is
extremely non-linear, and hence, to capture this non-linearity in our neural network, we
need it to incorporate non-linear functions that are capable of representing such
phenomena. 

By doing so, we increase the capacity of our neuron to model more complex
patterns that actually exist in the real world, and draw decision boundaries that would not
be possible, were we to only use linear functions. 

These types of functions, used to model
non-linear relationships in our data, are known as activation functions.

If the neurons don't have activation functions,
their output would be the weighted sum of the inputs, which is a linear function.
Then the entire neural network, that is, a composition of neurons, becomes a composition of
linear functions, which is also a linear function. 

This means that even if we add hidden
layers, the network will still be equivalent to a simple linear regression model, with all its
limitations. To turn the network into a non-linear function, we'll use non-linear activation
functions for the neurons. Usually, all neurons in the same layer have the same
activation function, but different layers may have different activation functions.

As with everything else in neural networks, you don’t have just one activation
function. You use the activation function that works best in a particular scenario.
With this in mind, you can break the activation functions into these categories:
**Step, Linear, Sigmoid, Tanh, ReLU, ELU, PReLU, LeakyReLU**