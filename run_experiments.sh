#!/bin/bash

DATA_DIR="$PWD/experiment-data-`date -I`"

if [ -d $DATA_DIR ] 
then
  echo "Directory $PWD/experiment-data exists.  Quitting not too overwrite existing results." 
else
  # dnnl nchw conv
  mkdir -p $DATA_DIR/dnnl_nchw_conv
  mkdir build-conv-nchw
  pushd build-conv-nchw
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_nchw_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11
  sudo make enable_turbo_boost
  make -j traffic
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_nchw_conv/
  popd

  # dnnl blocked conv
  mkdir -p $DATA_DIR/dnnl_blocked_conv
  mkdir build-conv-blocked
  pushd build-conv-blocked
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_blocked_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11
  sudo make enable_turbo_boost
  make -j traffic
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_blocked_conv/
  popd

  # dnnl tnc layer norm
  mkdir -p $DATA_DIR/dnnl_tnc_layer_norm
  mkdir build-tnc-layer_norm
  pushd build-tnc-layer_norm
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_tnc_layer_norm -DN=256 -DC=768 -DW=4 -DH=32
  sudo make enable_turbo_boost
  make -j traffic
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_tnc_layer_norm/
  popd
fi
