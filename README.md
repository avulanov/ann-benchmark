# ann-benchmark
Benchmarks of artificial neural network implementation for Spark MLlib https://github.com/avulanov/spark/tree/ann-interface-gemm. 

## Introduction
The goal is to benchmark the library, compare it with the other tools and test scalability with the number of nodes in the cluster.
  - Dataset: 
    - MNIST handwritten digits http://yann.lecun.com/exdb/mnist/
    - 60000 training set, 10000 test set 
    - Format: depends on the tool
  - Network: 
    - 6-layer NN 784-2500-2000-1500-1000-500-10 http://arxiv.org/pdf/1003.0358v1.pdf
    - ~12M weights total
  - Metrics:
    - Time needed for one epoch including gradient update 

The intention is to test a big model. Data is small so the time needed to read the data can be ignored. 

## Prerequisites
### OpenBLAS
  - Download, compile and install OpenBLAS https://github.com/xianyi/OpenBLAS
  - Add OpenBLAS to your library path:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/openblas
```
  - Create symlink to OpenBLAS within its folder
```
ln -s libopenblas.so libblas.so.3
```
  - You need to have gcc libs not less than v.4.8 in your library path `gcc -v`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/openblas:/gcc-4.8.2/lib
```

## Benchmark
### Spark
  - Clone Spark from https://github.com/avulanov/spark/tree/ann-interface-gemm
  - Compile Spark with `-Pnetlib-lgpl` flag to use native BLAS
  - Deploy Spark on N-node cluster
  - Download mnist dataset: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2 

### Caffe
  - Download Caffe
  - Configure `Makefile.config` to point to the same OpenBLAS lib (and CUDA) as for Spark
  - Change Solver to Double precision `tools/caffe.cpp`:
```
void CopyLayers(caffe::Solver<double>* solver, const std::string& model_list) {
  shared_ptr<caffe::Solver<double> >
    solver(caffe::GetSolver<double>(solver_param));
```
  - Compile Caffe
  - Set environment variable:
```    
export $CAFFE=/your/caffe/
```
  - Download mnist dataset and convert it to lmdb: 
```
$CAFFE/data/mnist/get_mnist.sh
$CAFFE/examples/mnist/create_mnist.sh
```
  - Run the benchmark with the model provided:
```
$CAFFE/build/tools/caffe train --solver=mnist-lmdb-5h.solver
```
## Benchmark
  - Hardware/Software setup:
    - Intel(R) Xeon(R) CPU E31240 @ 3.30GHz, 16GB RAM 
    - RedHat 6.4, OpenBLAS
    - Total 6 machines
    - Spark: one master, 5 workers

Preliminary results (s):

Nodes	| ANN-total	| ANN-compute	| Caffe	| Caffe60K |
------|-----------|-------------|-------|----------|
5 |	29 |	21 |	62 | 56 |
4	| 27 |	24	| 62	| 56 |
3 |	35.2 |	33 |	62 |	56 |
2	| 47 |	44	| 62	| 56 |
1	| 86	| 84	| 62	| 56 |

