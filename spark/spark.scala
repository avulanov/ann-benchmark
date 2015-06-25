import org.apache.log4j._
Logger.getRootLogger.setLevel(Level.OFF)
import org.apache.spark.mllib.ann.{FeedForwardTrainer, FeedForwardTopology}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.ANNClassifier
// huge topology
val train = MLUtils.loadLibSVMFile(sc, "hdfs://my.server:9000/data/mnist.scale")
val train5 = train.repartition(5)
train5.persist
train5.count
val topology = FeedForwardTopology.multiLayerPerceptron(Array[Int](780, 2500, 2000, 1500, 1000, 500, 10), false)
val trainer = new FeedForwardTrainer(topology, 780, 10).setBatchSize(1000)
trainer.SGDOptimizer.setNumIterations(2).setMiniBatchFraction(1.0).setStepSize(0.03)
val model = new ANNClassifier(trainer).train(train5)
