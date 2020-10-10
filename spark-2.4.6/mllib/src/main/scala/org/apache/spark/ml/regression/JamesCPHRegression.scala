/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package org.apache.spark.ml.regression

import scala.collection.mutable

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import org.apache.hadoop.fs.Path

import org.apache.spark.SparkException
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.storage.StorageLevel


private[regression] trait JamesCPHRegressionParams extends Params
  with HasFeaturesCol with HasLabelCol with HasPredictionCol with HasMaxIter
  with HasTol with HasFitIntercept with HasAggregationDepth with Logging {
  /**
   * Param for censor column name.
   * The value of this column could be 0 or 1.
   * If the value is 1, it means the event has occurred i.e. uncensored; otherwise censored.
   *
   * @group param
   */
  @Since("1.6.0")
  final val censorCol: Param[String] = new Param(this, "censorCol", "censor column name")

  /** @group getParam */
  @Since("1.6.0")
  def getCensorCol: String = $(censorCol)

  setDefault(censorCol -> "censor")

  /**
   * Param for quantile probabilities array.
   * Values of the quantile probabilities array should be in the range (0, 1)
   * and the array should be non-empty.
   *
   * @group param
   */
  @Since("1.6.0")
  final val quantileProbabilities: DoubleArrayParam = new DoubleArrayParam(this,
    "quantileProbabilities", "quantile probabilities array",
    (t: Array[Double]) => t.forall(ParamValidators.inRange(0, 1, false, false)) && t.length > 0)

  /** @group getParam */
  @Since("1.6.0")
  def getQuantileProbabilities: Array[Double] = $(quantileProbabilities)

  setDefault(quantileProbabilities -> Array(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))

  /**
   * Param for quantiles column name.
   * This column will output quantiles of corresponding quantileProbabilities if it is set.
   *
   * @group param
   */
  @Since("1.6.0")
  final val quantilesCol: Param[String] = new Param(this, "quantilesCol", "quantiles column name")

  /** Checks whether the input has quantiles column name. */
  private[regression] def hasQuantilesCol: Boolean = {
    isDefined(quantilesCol) && $(quantilesCol).nonEmpty
  }

  /**
   * Validates and transforms the input schema with the provided param map.
   *
   * @param schema  input schema
   * @param fitting whether this is in fitting or prediction
   * @return output schema
   */
  protected def validateAndTransformSchema(
                                            schema: StructType,
                                            fitting: Boolean): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    if (fitting) {
      SchemaUtils.checkNumericType(schema, $(censorCol))
      SchemaUtils.checkNumericType(schema, $(labelCol))
    }

    val schemaWithQuantilesCol = if (hasQuantilesCol) {
      SchemaUtils.appendColumn(schema, $(quantilesCol), new VectorUDT)
    } else schema

    SchemaUtils.appendColumn(schemaWithQuantilesCol, $(predictionCol), DoubleType)
  }
}


/**
 * :: Experimental ::
 * Model produced by [[JamesCPHRegression]].
 */
@Experimental
@Since("1.6.0")
class JamesCPHRegressionModel private[ml](
                                           @Since("1.6.0") override val uid: String,
                                           @Since("2.0.0") val coefficients: Vector,
                                           @Since("1.6.0") val intercept: Double,
                                           @Since("1.6.0") val scale: Double)
  extends Model[JamesCPHRegressionModel] with JamesCPHRegressionParams with MLWritable {

  override def copy(extra: ParamMap): JamesCPHRegressionModel = {
    copyValues(new JamesCPHRegressionModel(uid, coefficients, intercept, scale), extra)
      .setParent(parent)
  }

  /**
   * Returns an `MLWriter` instance for this ML instance.
   */
  override def write: MLWriter =
    new JamesCPHRegressionModel.JamesCPHRegressionModelWriter(this)

  @Since("2.0.0")
  def predict(features: Vector): Double = {
    math.exp(BLAS.dot(coefficients, features) + intercept)
  }

  @Since("2.0.0")
  def predictQuantiles(features: Vector): Vector = {
    // scale parameter for the Weibull distribution of lifetime
    val lambda = math.exp(BLAS.dot(coefficients, features) + intercept)
    // shape parameter for the Weibull distribution of lifetime
    val k = 1 / scale
    val quantiles = $(quantileProbabilities).map {
      q => lambda * math.exp(math.log(-math.log(1 - q)) / k)
    }
    Vectors.dense(quantiles)
  }

  /**
   * Transforms the input dataset.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val predictUDF = udf { features: Vector => predict(features) }
    val predictQuantilesUDF = udf { features: Vector => predictQuantiles(features) }
    if (hasQuantilesCol) {
      dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
        .withColumn($(quantilesCol), predictQuantilesUDF(col($(featuresCol))))
    } else {
      dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    }
  }

  /**
   * :: DeveloperApi ::
   *
   * Check transform validity and derive the output schema from the input schema.
   *
   * We check validity for interactions between parameters during `transformSchema` and
   * raise an exception if any parameter value is invalid. Parameter value checks which
   * do not depend on other parameters are handled by `Param.validate()`.
   *
   * Typical implementation should first conduct verification on schema change and parameter
   * validity, including complex parameter interaction checks.
   */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = false)
  }
}

class JamesCPHRegression(@Since("1.6.0") override val uid: String)
  extends Estimator[JamesCPHRegressionModel] with JamesCPHRegressionParams
    with DefaultParamsWritable with Logging {

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  @Since("1.6.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  @Since("1.6.0")
  def setTol(value: Double): this.type = set(tol, value)

  setDefault(tol -> 1E-6)

  /**
   * Suggested depth for treeAggregate (greater than or equal to 2).
   * If the dimensions of features or the number of partitions are large,
   * this param could be adjusted to a larger size.
   * Default is 2.
   *
   * @group expertSetParam
   */
  @Since("2.1.0")
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  setDefault(aggregationDepth -> 2)

  @Since("1.6.0")
  def this() = this(Identifiable.randomUID("aftSurvReg"))

  /** @group setParam */
  @Since("1.6.0")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setCensorCol(value: String): this.type = set(censorCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setQuantileProbabilities(value: Array[Double]): this.type = set(quantileProbabilities, value)

  /** @group setParam */
  @Since("1.6.0")
  def setQuantilesCol(value: String): this.type = set(quantilesCol, value)

  /**
   * Extract [[featuresCol]], [[labelCol]] and [[censorCol]] from input dataset,
   * and put it in an RDD with strong types.
   */
  protected[ml] def extractCPHPoints(dataset: Dataset[_]): RDD[CPHPoint] = {
    dataset.select(col($(featuresCol)), col($(labelCol)).cast(DoubleType),
      col($(censorCol)).cast(DoubleType)).rdd.map {
      case Row(features: Vector, label: Double, censor: Double) =>
        CPHPoint(features, label, censor)
    }
  }

  /**
   * Fits a model to the input data.
   */
  override def fit(dataset: Dataset[_]): JamesCPHRegressionModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)
    val instances = extractCPHPoints(dataset)
    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val featuresSummarizer = {
      val seqOp = (c: MultivariateOnlineSummarizer, v: CPHPoint) => c.add(v.features)
      val combOp = (c1: MultivariateOnlineSummarizer, c2: MultivariateOnlineSummarizer) => {
        c1.merge(c2)
      }
      instances.treeAggregate(
        new MultivariateOnlineSummarizer
      )(seqOp, combOp, $(aggregationDepth))
    }

    val featuresStd = featuresSummarizer.variance.toArray.map(math.sqrt)
    val numFeatures = featuresStd.size // 2
    println(">>> featuresStd")
    println(featuresStd)

    // Instrumentation: [736233b2] Stage class: JamesCPHRegression
    // Instrumentation: [736233b2] Stage uid: CPHSurvReg_8ea60eeb808e
    instr.logPipelineStage(this)
    // Instrumentation: [736233b2] training: numPartitions=1 storageLevel=StorageLevel(1 replicas)
    instr.logDataset(dataset)
    // Instrumentation: [736233b2] {"quantilesCol":"quantiles"}
    instr.logParams(this, labelCol, featuresCol, censorCol, predictionCol, quantilesCol,
      fitIntercept, maxIter, tol, aggregationDepth)
    // Instrumentation: [736233b2] {"quantileProbabilities.size":2}
    instr.logNamedValue("quantileProbabilities.size", $(quantileProbabilities).length)
    // Instrumentation: [736233b2] {"numFeatures":2}
    instr.logNumFeatures(numFeatures)
    // Instrumentation: [736233b2] {"numExamples":5}
    instr.logNumExamples(featuresSummarizer.count)

    if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
      featuresStd(i) == 0.0 && featuresSummarizer.mean(i) != 0.0
    }) {
      instr.logWarning("Fitting JamesCPHRegressionModel without intercept on dataset with " +
        "constant nonzero column, Spark MLlib outputs zero coefficients for constant nonzero " +
        "columns. This behavior is different from R survival::survreg.")
    }

    // Broadcast featuresStd
    val bcFeaturesStd = instances.context.broadcast(featuresStd)

    // TODO
    /**
     * create by james on 2020-10-03.
     *
     * costFun = new CPHCostFun
     * optimizer = new BreezeLBFGS
     */
    val costFun = new CPHCostFun(instances, $(fitIntercept), bcFeaturesStd, $(aggregationDepth))
    val optimizer = new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    println(">>> optimizer")
    println($(fitIntercept)) // true
    println($(aggregationDepth)) // 2
    println($(maxIter)) // 100
    println($(tol)) // 1.0E-6

    /*
       The parameters vector has three parts:
       the first element: Double, log(sigma), the log of scale parameter
       the second element: Double, intercept of the beta parameter
       the third to the end elements: Doubles, regression coefficients vector of the beta parameter
     */
    val initialParameters = Vectors.zeros(numFeatures + 2)

    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initialParameters.asBreeze.toDenseVector)
    println(">>> states")
    // println(states.size)

    // TODO
    /**
     * create by james on 2020-10-03.
     *
     * parameters 是训练的结果
     */
    println(">>> parameters")
    val parameters = {
      val arrayBuilder = mutable.ArrayBuilder.make[Double]
      var state: optimizer.State = null
      while (states.hasNext) {
        // Get the newest state as parameters
        state = states.next()
        //        arrayBuilder += state.adjustedValue
        //        println("state.adjustedValue = " + state.adjustedValue)
      }
      if (state == null) {
        val msg = s"${optimizer.getClass.getName} failed."
        throw new SparkException(msg)
      }
      state.x.toArray.clone()
    }

    bcFeaturesStd.destroy(blocking = false)
    if (handlePersistence) instances.unpersist()

    println("parameters.size = " + parameters.size) // 2
    val rawCoefficients = parameters.slice(2, parameters.length)
    var i = 0
    while (i < numFeatures) {
      rawCoefficients(i) *= {
        if (featuresStd(i) != 0.0) 1.0 / featuresStd(i) else 0.0
      }
      i += 1
    }
    val coefficients = Vectors.dense(rawCoefficients)
    val intercept = parameters(1)
    val scale = math.exp(parameters(0))
    copyValues(new JamesCPHRegressionModel(uid, coefficients, intercept, scale).setParent(this))
  }

  @Since("1.6.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema, fitting = true)
  }

  @Since("1.6.0")
  override def copy(extra: ParamMap): JamesCPHRegression = defaultCopy(extra)
}

@Since("1.6.0")
object JamesCPHRegressionModel extends MLReadable[JamesCPHRegressionModel] {

  @Since("1.6.0")
  override def read: MLReader[JamesCPHRegressionModel] = new JamesCPHRegressionModelReader

  @Since("1.6.0")
  override def load(path: String): JamesCPHRegressionModel = super.load(path)

  /** [[MLWriter]] instance for [[JamesCPHRegressionModel]] */
  private[JamesCPHRegressionModel] class JamesCPHRegressionModelWriter(
                                                                              instance: JamesCPHRegressionModel
                                                                            ) extends MLWriter with Logging {

    private case class Data(coefficients: Vector, intercept: Double, scale: Double)

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: coefficients, intercept, scale
      val data = Data(instance.coefficients, instance.intercept, instance.scale)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class JamesCPHRegressionModelReader extends MLReader[JamesCPHRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[JamesCPHRegressionModel].getName

    override def load(path: String): JamesCPHRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(coefficients: Vector, intercept: Double, scale: Double) =
        MLUtils.convertVectorColumnsToML(data, "coefficients")
          .select("coefficients", "intercept", "scale")
          .head()
      val model = new JamesCPHRegressionModel(metadata.uid, coefficients, intercept, scale)

      metadata.getAndSetParams(model)
      model
    }
  }
}


private class CPHAggregator(
                             bcParameters: Broadcast[BDV[Double]],
                             fitIntercept: Boolean,
                             bcFeaturesStd: Broadcast[Array[Double]]) extends Serializable {

  private val length = bcParameters.value.length
  // make transient so we do not serialize between aggregation stages
  @transient private lazy val parameters = bcParameters.value
  // the regression coefficients to the covariates
  @transient private lazy val coefficients = parameters.slice(2, length)
  @transient private lazy val intercept = parameters(1)
  // sigma is the scale parameter of the CPH model
  @transient private lazy val sigma = math.exp(parameters(0))

  private var totalCnt: Long = 0L
  private var lossSum = 0.0
  // Here we optimize loss function over log(sigma), intercept and coefficients
  private lazy val gradientSumArray = Array.ofDim[Double](length)

  def count: Long = totalCnt

  def loss: Double = {
    require(totalCnt > 0.0, s"The number of instances should be " +
      s"greater than 0.0, but got $totalCnt.")
    lossSum / totalCnt
  }

  def gradient: BDV[Double] = {
    require(totalCnt > 0.0, s"The number of instances should be " +
      s"greater than 0.0, but got $totalCnt.")
    new BDV(gradientSumArray.map(_ / totalCnt.toDouble))
  }

  // TODO
  /**
   * create by james on 2020-10-03.
   *
   *org.apache.spark.ml.regression.CPHAggregator#add(org.apache.spark.ml.regression.CPHPoint)
   */
  /**
   * Add a new training data to this CPHAggregator, and update the loss and gradient
   * of the objective function.
   *
   * @param data The CPHPoint representation for one data point to be added into this aggregator.
   * @return This CPHAggregator object.
   */
  def add(data: CPHPoint): this.type = {
    val xi = data.features
    val ti = data.label
    val delta = data.censor

    require(ti > 0.0, "The lifetime or label should be  greater than 0.")

    val localFeaturesStd = bcFeaturesStd.value

    val margin = {
      var sum = 0.0
      xi.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += coefficients(index) * (value / localFeaturesStd(index))
        }
      }
      sum + intercept
    }
    val epsilon = (math.log(ti) - margin) / sigma

    lossSum += delta * math.log(sigma) - delta * epsilon + math.exp(epsilon)

    val multiplier = (delta - math.exp(epsilon)) / sigma

    gradientSumArray(0) += delta + multiplier * sigma * epsilon
    gradientSumArray(1) += {
      if (fitIntercept) multiplier else 0.0
    }
    xi.foreachActive { (index, value) =>
      if (localFeaturesStd(index) != 0.0 && value != 0.0) {
        gradientSumArray(index + 2) += multiplier * (value / localFeaturesStd(index))
      }
    }

    totalCnt += 1
    this
  }

  // TODO
  /**
   * create by james on 2020-10-03.
   *
   * org.apache.spark.ml.regression.CPHAggregator#merge(org.apache.spark.ml.regression.CPHAggregator)
   */
  /**
   * Merge another CPHAggregator, and update the loss and gradient
   * of the objective function.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other CPHAggregator to be merged.
   * @return This CPHAggregator object.
   */
  def merge(other: CPHAggregator): this.type = {
    if (other.count != 0) {
      totalCnt += other.totalCnt
      lossSum += other.lossSum

      var i = 0
      while (i < length) {
        this.gradientSumArray(i) += other.gradientSumArray(i)
        i += 1
      }
    }
    this
  }
} // class CPHAggregator


// TODO
/**
 * create by james on 2020-10-03.
 *
 * class CPHCostFun
 */
/**
 * CPHCostFun implements Breeze's DiffFunction[T] for CPH cost.
 * It returns the loss and gradient at a particular point (parameters).
 * It's used in Breeze's convex optimization routines.
 */
private class CPHCostFun(
                          data: RDD[CPHPoint],
                          fitIntercept: Boolean,
                          bcFeaturesStd: Broadcast[Array[Double]],
                          aggregationDepth: Int) extends DiffFunction[BDV[Double]] {

  // TODO
  /**
   * create by james on 2020-10-03.
   *
   * org.apache.spark.ml.regression.CPHCostFun#calculate(breeze.linalg.DenseVector)
   */
  override def calculate(parameters: BDV[Double]): (Double, BDV[Double]) = {

    val bcParameters = data.context.broadcast(parameters)

    val CPHAggregator = data.treeAggregate(
      new CPHAggregator(bcParameters, fitIntercept, bcFeaturesStd))(
      seqOp = (c, v) => (c, v) match {
        case (aggregator, instance) => aggregator.add(instance)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case (aggregator1, aggregator2) => aggregator1.merge(aggregator2)
      }, depth = aggregationDepth)

    bcParameters.destroy(blocking = false)
    (CPHAggregator.loss, CPHAggregator.gradient)
  }
} // class CPHCostFun

/**
 * Class that represents the (features, label, censor) of a data point.
 *
 * @param features List of features for this data point.
 * @param label    Label for this data point.
 * @param censor   Indicator of the event has occurred or not. If the value is 1, it means
 *                 the event has occurred i.e. uncensored; otherwise censored.
 */
private[regression] case class CPHPoint(features: Vector, label: Double, censor: Double) {
  require(censor == 1.0 || censor == 0.0, "censor of class CPHPoint must be 1.0 or 0.0")
} // class CPHPoint
