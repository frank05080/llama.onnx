Note that find_package(pybind11) will only work correctly if pybind11 has been correctly installed on the system, e. g. after downloading or cloning the pybind11 repository :

# Classic CMake
cd pybind11
mkdir build
cd build
cmake ..
make -j4
sudo make install

# CMake 3.15+
cd pybind11
cmake -S . -B build
cmake --build build -j 2  # Build on 2 cores
cmake --install build

## Sample CMakeLists.txt

cmake_minimum_required(VERSION 3.4...3.18)
project(pybind11_sample LANGUAGES CXX)

find_package(pybind11 REQUIRED)
pybind11_add_module(pybind11_sample main.cpp)

## To use sample, do the following:

export PYTHONPATH=$PYTHONPATH:/root/rwkv/pybind_test/build
export PYTHONPATH=$PYTHONPATH:/root/rwkv/pybind_infer/build