import org.apache.log4j._
Logger.getRootLogger.setLevel(Level.OFF)
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

// maximum number of worker nodes in cluster
val numNodes = 5
// batch size, ~10K is good for GPU
val batchSize = 1000
// number of iterations to run
val numIterations = 5
val train = MLUtils.loadLibSVMFile(sc, "file:///data/mnist/mnist.scale")
//val layers = Array[Int](780, 2500, 2000, 1500, 1000, 500, 10)
val layers = Array[Int](780, 10)
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(1000).setSeed(1234L).setMaxIter(1)
for (i <- 1 to numNodes) {
  val dataPartitions = sc.parallelize(1 to i, i)
  val sample = train.sample(true, 1.0 / i, 11L).collect
  val parallelData = sqlContext.createDataFrame(dataPartitions.flatMap(x => sample))
  parallelData.persist
  parallelData.count
  val t = System.nanoTime()
  val model = trainer.fit(parallelData)
  println(i + "\t" + batchSize + "\t" + (System.nanoTime() - t) / (numIterations * 1e9)) 	
  parallelData.unpersist()
}
