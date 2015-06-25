# ann-benchmark
Benchmarks of artificial neural network library for Spark MLlib

## spark
Execute 

## caffe
  - Download and compile Caffe
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
