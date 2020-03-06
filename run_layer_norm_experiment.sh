#!/bin/bash

# Target_data_dir, local dir, algo, cache, threading
function layer_norm_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  cmake ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=768 -DW=4 -DH=32 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}


DATA_DIR="$PWD/experiment-layer_norm-`date -I`"

if [ -d $DATA_DIR ] 
then
  echo "Directory $PWD/experiment-layer_norm-`date -I` exists.  Quitting not too overwrite existing results." 
else
  # SINGLE THREAD
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_warm_caches-single build-tnc-layer_norm_warm_caches-single dnnl_tnc_layer_norm false single
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_cold_caches-single build-tnc-layer_norm_cold_caches-single dnnl_tnc_layer_norm true single
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_warm_caches-single build-tnc-layer_norm_warm_caches-single dnnl_tnc_layer_norm_inplace false single
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_cold_caches-single build-tnc-layer_norm_cold_caches-single dnnl_tnc_layer_norm_inplace true single
fi
