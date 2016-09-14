package org.imshealth.ml.evaluation

import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{Identifiable}
import org.imshealth.mllib.evaluation.BinaryClassificationMetricsIMSPA
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.sql.types.{DataType, StructType}

/**
 * :: Experimental ::
 * Evaluator for binary classification, which expects two input columns: rawPrediction and label.
 */
@Experimental
class BinaryClassificationEvaluatorIMSPA(override val uid: String)
  extends Evaluator with HasRawPredictionCol with HasLabelCol {

  def this() = this(Identifiable.randomUID("binEval"))

  /**
   * param for metric name in evaluation
   * Default: areaUnderROC
   * @group param
   */
  val metricName: Param[String] = {
    val allowedParams = ParamValidators.inArray(Array("areaUnderROC", "areaUnderPR", "pGivenR"))
    new Param(
      this, "metricName", "metric name in evaluation (areaUnderROC|areaUnderPR|pGivenR)", allowedParams)
  }

  val metricValue: Param[String] = {
    new Param(
      this, "metricValue", "metric value in evaluation (pGivenR)")
  }

  /** @group getParam */
  def getMetricName: String = $(metricName)

  /** @group setParam */
  def setMetricName(value: String): this.type = set(metricName, value)

  /** @group getParam */
  def getMetricValue: String = $(metricValue)

  /** @group setParam */
  def setMetricValue(value: String): this.type = set(metricValue, value)

  /** @group setParam */
  def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)

  /**
   * @group setParam
   * @deprecated use [[setRawPredictionCol()]] instead
   */
  @deprecated("use setRawPredictionCol instead", "1.5.0")
  def setScoreCol(value: String): this.type = set(rawPredictionCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  setDefault(metricName -> "areaUnderROC")
  setDefault(metricValue -> "0.5")

  override def evaluate(dataset: DataFrame): Double = {
    val schema = dataset.schema
    checkColumnType(schema, $(rawPredictionCol), new VectorUDT)
    checkColumnType(schema, $(labelCol), DoubleType)

    // TODO: When dataset metadata has been implemented, check rawPredictionCol vector length = 2.
    val scoreAndLabels = dataset.select($(rawPredictionCol), $(labelCol))
      .map { case Row(rawPrediction: Vector, label: Double) =>
        (rawPrediction(1), label)
      }
    val metricDouble = ($(metricValue)).toDouble
    val metrics = new BinaryClassificationMetricsIMSPA(scoreAndLabels)
    val metric = $(metricName) match {
      case "areaUnderROC" => metrics.areaUnderROC()
      case "areaUnderPR" => metrics.areaUnderPR()
      case "pGivenR" => metrics.pGivenR(metricDouble)
    }
    metrics.unpersist()
    metric
  }

  override def isLargerBetter: Boolean = $(metricName) match {
    case "areaUnderROC" => true
    case "areaUnderPR" => true
    case "pGivenR" => true
  }

  override def copy(extra: ParamMap): BinaryClassificationEvaluatorIMSPA = defaultCopy(extra)

  def checkColumnType(
                       schema: StructType,
                       colName: String,
                       dataType: DataType,
                       msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.equals(dataType),
      s"Column $colName must be of type $dataType but was actually $actualDataType.$message")
  }
}

trait HasRawPredictionCol extends Params {

  /**
    * Param for raw prediction (a.k.a. confidence) column name.
    * @group param
    */
  final val rawPredictionCol: Param[String] = new Param[String](this, "rawPredictionCol", "raw prediction (a.k.a. confidence) column name")

  setDefault(rawPredictionCol, "rawPrediction")

  /** @group getParam */
  final def getRawPredictionCol: String = $(rawPredictionCol)
}

trait HasLabelCol extends Params {

  /**
    * Param for label column name.
    * @group param
    */
  final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

  setDefault(labelCol, "label")

  /** @group getParam */
  final def getLabelCol: String = $(labelCol)
}
