#!/bin/bash
echo "Hello!, I will just execute a couple of commands to make sure the system will work"

echo "Installing dependencies"
sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev

echo "Installing OpenCV"
sudo apt-get install python-opencv

echo "Installing libfreenect"
sudo apt-get install libfreenect-dev freenect python-freenect
sudo apt-get install cython
sudo apt-get install python-dev
sudo apt-get install python-numpy
sudo apt-get install python-matplotlib
