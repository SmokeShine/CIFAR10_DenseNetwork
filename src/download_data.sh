#!/bin/bash

echo -n "Downloading Training Data"
wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo -n "Unzipping Data"
tar -xvf cifar-10-python.tar.gz 
