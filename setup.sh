#!/bin/bash
echo "Hello!, I will just execute a couple of commands to make sure the system will work"

echo "Installing dependencies"
sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev

echo "Installing libfreenect"
mkdir ~/Kinect
cd ~/Kinect
git clone git://github.com/OpenKinect/libfreenect.git
cd libfreenect
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib64/
cd ..

echo "Installing OpenCV"
git clone https://github.com/Itseez/opencv.git
cd ~/opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
cd ..
