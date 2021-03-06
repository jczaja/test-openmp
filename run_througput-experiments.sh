#!/bin/bash

CMAKE_COMMAND=cmake  # change this if needed

# Target_data_dir, local dir, algo, cache, threading
function conv_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=16 -DW=227 -DH=227 -DNF=96 -DHF=3 -DWF=3 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}

# Target_data_dir, local dir, algo, cache, threading
function layer_norm_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=dnnl_tnc_layer_norm -DN=256 -DC=768 -DW=4 -DH=32 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}

function eltwise_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=96 -DW=55 -DH=55 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}

function binary_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=96 -DW=55 -DH=55 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}

function pool_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=96 -DW=55 -DH=55 -DWF=5 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}

function inner_prod_experiment()
{
  mkdir -p $1
  mkdir $2
  pushd $2
  $CMAKE_COMMAND ../ -DCMAKE_BUILD_TYPE=Release -DALGO=$3 -DN=256 -DC=4096 -DW=1 -DH=1 -DNF=1000 -DCOLD_CACHES=$4 -DTHREADING=$5 -DCHARTS=relative
  sudo make enable_turbo_boost
  make -j 4 test-openmp-gomp
  sudo make disable_turbo_boost
  make roofline
  cp roofline* cputest.txt memtest*.txt runtime.txt traffic.txt work.txt algo_info.txt cpu_info.txt $1
  popd
}



DATA_DIR="$PWD/experiment-throughput-`date -I`"

if [ -d $DATA_DIR ] 
then
  echo "Directory $PWD/experiment-throughput-`date -I` exists.  Quitting not too overwrite existing results." 
else
  # FULL SOCKETS
  conv_experiment $DATA_DIR/dnnl_nchw_conv_warm_caches build-conv-nchw-warm-caches dnnl_nchw_conv false full
  conv_experiment $DATA_DIR/dnnl_nchw_conv_cold_caches build-conv-nchw-cold-caches dnnl_nchw_conv true full
  conv_experiment $DATA_DIR/dnnl_blocked_conv_warm_caches build-conv-blocked-warm-caches dnnl_blocked_conv false full
  conv_experiment $DATA_DIR/dnnl_blocked_conv_cold_caches build-conv-blocked-cold-caches dnnl_blocked_conv true full
  conv_experiment $DATA_DIR/dnnl_winograd_conv_warm_caches build-conv-winograd-warm-caches dnnl_winograd_conv false full
  conv_experiment $DATA_DIR/dnnl_winograd_conv_cold_caches build-conv-winograd-cold-caches dnnl_winograd_conv true full
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_warm_caches build-tnc-layer_norm_warm_caches dnnl_tnc_layer_norm false full
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_cold_caches build-tnc-layer_norm_cold_caches dnnl_tnc_layer_norm true full
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_warm_caches build-eltwise-nchw-warm-caches dnnl_nchw_gelu false full
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_cold_caches build-eltwise-nchw-cold-caches dnnl_nchw_gelu true full
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_warm_caches build-eltwise-blocked-warm-caches dnnl_blocked_gelu false full
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_cold_caches build-eltwise-blocked-cold-caches dnnl_blocked_gelu true full
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_warm_caches build-binary-add-nchw-warm-caches dnnl_nchw_binary_add false full
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_cold_caches build-binary-add-nchw-cold-caches dnnl_nchw_binary_add true full
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_warm_caches build-binary-add-blocked-warm-caches dnnl_blocked_binary_add false full
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_cold_caches build-binary-add-blocked-cold-caches dnnl_blocked_binary_add true full
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_warm_caches build-binary-mul-nchw-warm-caches dnnl_nchw_binary_mul false full
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_cold_caches build-binary-mul-nchw-cold-caches dnnl_nchw_binary_mul true full
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_warm_caches build-binary-mul-blocked-warm-caches dnnl_blocked_binary_mul false full
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_cold_caches build-binary-mul-blocked-cold-caches dnnl_blocked_binary_mul true full
  pool_experiment $DATA_DIR/dnnl_nchw_pool_warm_caches build-pool-nchw-warm-caches dnnl_nchw_pool_avg false full
  pool_experiment $DATA_DIR/dnnl_nchw_pool_cold_caches build-pool-nchw-cold-caches dnnl_nchw_pool_avg true full
  pool_experiment $DATA_DIR/dnnl_blocked_pool_warm_caches build-pool-blocked-warm-caches dnnl_blocked_pool_avg false full
  pool_experiment $DATA_DIR/dnnl_blocked_pool_cold_caches build-pool-blocked-cold-caches dnnl_blocked_pool_avg true full
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_cold_caches build-fc-nchw-cold-caches dnnl_nchw_fc true full
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_warm_caches build-fc-nchw-warm-caches dnnl_nchw_fc false full


  # ONE SOCKET
  conv_experiment $DATA_DIR/dnnl_nchw_conv_warm_caches-socket build-conv-nchw-warm-caches-socket dnnl_nchw_conv false socket
  conv_experiment $DATA_DIR/dnnl_nchw_conv_cold_caches-socket build-conv-nchw-cold-caches-socket dnnl_nchw_conv true socket
  conv_experiment $DATA_DIR/dnnl_blocked_conv_warm_caches-socket build-conv-blocked-warm-caches-socket dnnl_blocked_conv false socket
  conv_experiment $DATA_DIR/dnnl_blocked_conv_cold_caches-socket build-conv-blocked-cold-caches-socket dnnl_blocked_conv true socket
  conv_experiment $DATA_DIR/dnnl_winograd_conv_warm_caches-socket build-conv-winograd-warm-caches-socket dnnl_winograd_conv false socket
  conv_experiment $DATA_DIR/dnnl_winograd_conv_cold_caches-socket build-conv-winograd-cold-caches-socket dnnl_winograd_conv true socket
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_warm_caches-socket build-tnc-layer_norm_warm_caches-socket dnnl_tnc_layer_norm false socket
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_cold_caches-socket build-tnc-layer_norm_cold_caches-socket dnnl_tnc_layer_norm true socket
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_warm_caches-socket build-eltwise-nchw-warm-caches-socket dnnl_nchw_gelu false socket
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_cold_caches-socket build-eltwise-nchw-cold-caches-socket dnnl_nchw_gelu true socket
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_warm_caches-socket build-eltwise-blocked-warm-caches-socket dnnl_blocked_gelu false socket
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_cold_caches-socket build-eltwise-blocked-cold-caches-socket dnnl_blocked_gelu true socket
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_warm_caches-socket build-binary-add-nchw-warm-caches-socket dnnl_nchw_binary_add false socket
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_cold_caches-socket build-binary-add-nchw-cold-caches-socket dnnl_nchw_binary_add true socket
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_warm_caches-socket build-binary-add-blocked-warm-caches-socket dnnl_blocked_binary_add false socket
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_cold_caches-socket build-binary-add-blocked-cold-caches-socket dnnl_blocked_binary_add true socket
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_warm_caches-socket build-binary-mul-nchw-warm-caches-socket dnnl_nchw_binary_mul false socket
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_cold_caches-socket build-binary-mul-nchw-cold-caches-socket dnnl_nchw_binary_mul true socket
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_warm_caches-socket build-binary-mul-blocked-warm-caches-socket dnnl_blocked_binary_mul false socket
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_cold_caches-socket build-binary-mul-blocked-cold-caches-socket dnnl_blocked_binary_mul true socket
  pool_experiment $DATA_DIR/dnnl_nchw_pool_warm_caches-socket build-pool-nchw-warm-caches-socket dnnl_nchw_pool_avg false socket
  pool_experiment $DATA_DIR/dnnl_nchw_pool_cold_caches-socket build-pool-nchw-cold-caches-socket dnnl_nchw_pool_avg true socket
  pool_experiment $DATA_DIR/dnnl_blocked_pool_warm_caches-socket build-pool-blocked-warm-caches-socket dnnl_blocked_pool_avg false socket
  pool_experiment $DATA_DIR/dnnl_blocked_pool_cold_caches-socket build-pool-blocked-cold-caches-socket dnnl_blocked_pool_avg true socket
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_cold_caches-socket build-fc-nchw-cold-caches-socket dnnl_nchw_fc true socket
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_warm_caches-socket build-fc-nchw-warm-caches-socket dnnl_nchw_fc false socket

  # SINGLE THREAD
  conv_experiment $DATA_DIR/dnnl_nchw_conv_warm_caches-single build-conv-nchw-warm-caches-single dnnl_nchw_conv false single
  conv_experiment $DATA_DIR/dnnl_nchw_conv_cold_caches-single build-conv-nchw-cold-caches-single dnnl_nchw_conv true single
  conv_experiment $DATA_DIR/dnnl_blocked_conv_warm_caches-single build-conv-blocked-warm-caches-single dnnl_blocked_conv false single
  conv_experiment $DATA_DIR/dnnl_blocked_conv_cold_caches-single build-conv-blocked-cold-caches-single dnnl_blocked_conv true single
  conv_experiment $DATA_DIR/dnnl_winograd_conv_warm_caches-single build-conv-winograd-warm-caches-single dnnl_winograd_conv false single
  conv_experiment $DATA_DIR/dnnl_winograd_conv_cold_caches-single build-conv-winograd-cold-caches-single dnnl_winograd_conv true single
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_warm_caches-single build-tnc-layer_norm_warm_caches-single dnnl_tnc_layer_norm false single
  layer_norm_experiment $DATA_DIR/dnnl_tnc_layer_norm_cold_caches-single build-tnc-layer_norm_cold_caches-single dnnl_tnc_layer_norm true single
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_warm_caches-single build-eltwise-nchw-warm-caches-single dnnl_nchw_gelu false single
  eltwise_experiment $DATA_DIR/dnnl_nchw_gelu_cold_caches-single build-eltwise-nchw-cold-caches-single dnnl_nchw_gelu true single
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_warm_caches-single build-eltwise-blocked-warm-caches-single dnnl_blocked_gelu false single
  eltwise_experiment $DATA_DIR/dnnl_blocked_gelu_cold_caches-single build-eltwise-blocked-cold-caches-single dnnl_blocked_gelu true single
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_warm_caches-single build-binary-add-nchw-warm-caches-single dnnl_nchw_binary_add false single
  binary_experiment $DATA_DIR/dnnl_nchw_binary_add_cold_caches-single build-binary-add-nchw-cold-caches-single dnnl_nchw_binary_add true single
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_warm_caches-single build-binary-add-blocked-warm-caches-single dnnl_blocked_binary_add false single
  binary_experiment $DATA_DIR/dnnl_blocked_binary_add_cold_caches-single build-binary-add-blocked-cold-caches-single dnnl_blocked_binary_add true single
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_warm_caches-single build-binary-mul-nchw-warm-caches-single dnnl_nchw_binary_mul false single
  binary_experiment $DATA_DIR/dnnl_nchw_binary_mul_cold_caches-single build-binary-mul-nchw-cold-caches-single dnnl_nchw_binary_mul true single
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_warm_caches-single build-binary-mul-blocked-warm-caches-single dnnl_blocked_binary_mul false single
  binary_experiment $DATA_DIR/dnnl_blocked_binary_mul_cold_caches-single build-binary-mul-blocked-cold-caches-single dnnl_blocked_binary_mul true single
  pool_experiment $DATA_DIR/dnnl_nchw_pool_cold_caches-single build-pool-nchw-cold-caches-single dnnl_nchw_pool_avg true single
  pool_experiment $DATA_DIR/dnnl_nchw_pool_warm_caches-single build-pool-nchw-warm-caches-single dnnl_nchw_pool_avg false single
  pool_experiment $DATA_DIR/dnnl_blocked_pool_cold_caches-single build-pool-blocked-cold-caches-single dnnl_blocked_pool_avg true single
  pool_experiment $DATA_DIR/dnnl_blocked_pool_warm_caches-single build-pool-blocked-warm-caches-single dnnl_blocked_pool_avg false single
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_cold_caches-single build-fc-nchw-cold-caches-single dnnl_nchw_fc true single
  inner_prod_experiment $DATA_DIR/dnnl_nchw_fc_warm_caches-single build-fc-nchw-warm-caches-single dnnl_nchw_fc false single
fi
