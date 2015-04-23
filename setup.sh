#!/bin/bash
echo "Hello!, I will just execute a couple of commands to make sure the system will work"

echo "Installing dependencies"
sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev

echo "Installing OpenCV"
git clone https://github.com/Itseez/opencv.git
cd ~/opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
cd ../..

echo "Installing libfreenect"
sudo apt-get install freenect
sudo apt-get install cython
sudo apt-get install python-dev
sudo apt-get install python-numpy
sudo apt-get install python-matplotlib

mkdir tmp
cd tmp
git clone https://github.com/OpenKinect/libfreenect.git
cd libfreenect/wrappers/python/
sudo python setup.py install
cd ../../../..
