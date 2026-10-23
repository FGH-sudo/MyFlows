cmake_minimum_required(VERSION 3.20)

project(myflows_native_cublas LANGUAGES CXX)

# native_cublas.dll on Windows, native_cublas.so on Linux; only the MF_API entry points are exported.
add_library(native_cublas SHARED native_cublas.cpp)
target_compile_features(native_cublas PRIVATE cxx_std_17)
set_target_properties(native_cublas PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
# dlopen/dlsym live in libdl before glibc 2.34; CMAKE_DL_LIBS is empty on Windows.
target_link_libraries(native_cublas PRIVATE ${CMAKE_DL_LIBS})
