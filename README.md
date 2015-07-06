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
### GCC 4.8.2
RedHat 6.x has an older GCC compiler that has libgfortran library that is incompatible with netlib-java wrappers. Check GCC version: `gcc -v`. New GCC should be ALWAYS in your path:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/new/gcc/lib64
```

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
  - To use OpenBLAS, add it to your library path. Make sure there is no other folder with `libblas.so.3` in your path.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/openblas
```
### CUDA and NVBLAS (for GPU)
  - Download and install the latest CUDA and GPU driver (usually comes with CUDA)
  - NB! you need the reference CBLAS library to run NVBLAS through Spark
    - RedHat and Fedora
      - Download and compile CBLAS as a shared library: http://avulanov.blogspot.com/2015/03/cblas-compilation-as-shared-library.html
      - Create symlink to CBLAS within its folder
`ln -s cblas_LINUX.so libblas.so.3`
    - Ubuntu etc. `sudo apt-get install blas`
      - Make sure that the installed library has CBLAS symbols `objdump -T libblas.so.3 | grep "cblas"`
  - NVBLAS needs configuration file to run
    - Create `nvblas.conf` file within `/your/cuda/lib64`. Copy the contents from: http://docs.nvidia.com/cuda/nvblas/#NVBLAS_CONFIG_FILE. Modify the log path within the file.
    - Add a variable
```
export NVBLAS_CONFIG_FILE=/your/cuda/lib64/nvblas.conf
```
  - To use NVBLAS, add CBLAS and CUDA to your library path. Also, preload NVBLAS symbols (it is better to do this right before launching Spark otherwise all your shell commands will go through NVBLAS causing errors). Make sure there is no other folder with `libblas.so.3` in your path.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/cblas:/your/cuda/lib64
export LD_PRELOAD=/your/cuda/lib64/libnvblas.so
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

The lastest results are in the spreadsheet: https://docs.google.com/spreadsheets/d/13U1fwF5-h90X-VeF01dOT-IlJtwYa1AsRtCBjAkSqKI/edit?usp=sharing

