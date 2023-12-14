# Service 3.2.2 for Rio Vena Digital Twin - District Network Production Economic Optimization
## Introduction
This repo contains all the necessary modules enabling the possibilty to optmize the costs of a Network
The service consists of two main modules:
    <ul>The Artificial Neural Network Machine Learning Model: this allows the possibility to predict the future consumptions of the Buildings</ul>
    <ul>The Optimizer: given a trained model, the optimizer tries to find the best possible solution in order to manage the consumptions of the Buildings</ul>
and one support module:
    <ul>utils.py</ul>
In the following sections all the modules will be extensively described

All this software is developed under the DigiBUILD European Commission Funded Horizon 2020 Project
## ANN Model
The first part of the service consist of an Artificial Neural Network Model, aimed at predicting the future consumption of the boiler. 
The model takes as input a matrix of dimension (N, 6)
## Optimizer

