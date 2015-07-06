import org.apache.log4j._
Logger.getRootLogger.setLevel(Level.OFF)
import org.apache.spark.mllib.ann.{FeedForwardTrainer, FeedForwardTopology}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.ANNClassifier
// maximum number of worker nodes in cluster
val numNodes = 5
// batch size, ~10K is good for GPU
val batchSize = 10000
// number of iterations to run
val numIterations = 5
val train = MLUtils.loadLibSVMFile(sc, "/mnist.scale")
val topology = FeedForwardTopology.multiLayerPerceptron(Array[Int](780, 2500, 2000, 1500, 1000, 500, 10), false)
val trainer = new FeedForwardTrainer(topology, 780, 10).setBatchSize(batchSize)
trainer.SGDOptimizer.setNumIterations(numIterations).setMiniBatchFraction(1.0).setStepSize(0.03)
// parallalize the data for N nodes, persist, run X iterations and print average time for each run
for (i <- 1 to numNodes) {
	val dataPartitions = sc.parallelize(1 to i, i)
	val sample = train.sample(true, 1.0 / i, 11L).collect
	val parallelData = dataPartitions.flatMap(x => sample)
	parallelData.persist
	parallelData.count
	val t = System.nanoTime()
	val model = new ANNClassifier(trainer).train(parallelData)
	println(i + "\t" + batchSize + "\t" + (System.nanoTime() - t) / (numIterations * 1e9)) 	
}
