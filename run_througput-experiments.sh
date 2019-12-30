#!/bin/bash

DATA_DIR="$PWD/experiment-throughput-`date -I`"

if [ -d $DATA_DIR ] 
then
  echo "Directory $PWD/experiment-throughput-`date -I` exists.  Quitting not too overwrite existing results." 
else
  # dnnl nchw conv , warmed caches
  mkdir -p $DATA_DIR/dnnl_nchw_conv_warm_caches
  mkdir build-conv-nchw_warm_caches
  pushd build-conv-nchw_warm_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_nchw_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_nchw_conv_warm_caches
  popd

  # dnnl nchw conv , cold caches
  mkdir -p $DATA_DIR/dnnl_nchw_conv_cold_caches
  mkdir build-conv-nchw_cold_caches
  pushd build-conv-nchw_cold_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_nchw_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11 -DCOLD_CACHES=true
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_nchw_conv_cold_caches
  popd

  # dnnl blocked conv , warm caches
  mkdir -p $DATA_DIR/dnnl_blocked_conv_warm_caches
  mkdir build-conv-blocked_warm_caches
  pushd build-conv-blocked_warm_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_blocked_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11 
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_blocked_conv_warm_caches
  popd

  # dnnl blocked conv , cold caches
  mkdir -p $DATA_DIR/dnnl_blocked_conv_cold_caches
  mkdir build-conv-blocked_cold_caches
  pushd build-conv-blocked_cold_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_blocked_conv -DN=256 -DC=3 -DW=227 -DH=227 -DNF=96 -DHF=11 -DWF=11 -DCOLD_CACHES=true
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_blocked_conv_cold_caches
  popd

  # dnnl tnc layer norm, warm caches
  mkdir -p $DATA_DIR/dnnl_tnc_layer_norm
  mkdir build-tnc-layer_norm_warm_caches
  pushd build-tnc-layer_norm_warm_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_tnc_layer_norm -DN=256 -DC=768 -DW=4 -DH=32
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_tnc_layer_norm/
  popd

  # dnnl tnc layer norm, cold caches
  mkdir -p $DATA_DIR/dnnl_tnc_layer_norm_cold_caches
  mkdir build-tnc-layer_norm_cold_caches
  pushd build-tnc-layer_norm_cold_caches
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_tnc_layer_norm -DN=256 -DC=768 -DW=4 -DH=32 -DCOLD_CACHES=true
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $DATA_DIR/dnnl_tnc_layer_norm_cold_caches
  popd
fi
