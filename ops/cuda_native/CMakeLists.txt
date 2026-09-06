cmake_minimum_required(VERSION 3.20)

project(myflows_native_cublas LANGUAGES CXX)

add_library(native_cublas SHARED native_cublas.cpp)
target_compile_features(native_cublas PRIVATE cxx_std_17)
set_target_properties(native_cublas PROPERTIES PREFIX "")
