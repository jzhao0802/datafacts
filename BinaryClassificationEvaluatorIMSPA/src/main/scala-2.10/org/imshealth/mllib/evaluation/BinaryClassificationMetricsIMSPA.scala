package org.imshealth.mllib.evaluation

import org.apache.spark.Logging
import org.imshealth.mllib.evaluation.binary._
import org.apache.spark.rdd.{RDD, UnionRDD}
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Row}

/**
 * :: Experimental ::
 * Evaluator for binary classification.
 *
 * @param scoreAndLabels an RDD of (score, label) pairs.
 * @param numBins if greater than 0, then the curves (ROC curve, PR curve) computed internally
 *                will be down-sampled to this many "bins". If 0, no down-sampling will occur.
 *                This is useful because the curve contains a point for each distinct score
 *                in the input, and this could be as large as the input itself -- millions of
 *                points or more, when thousands may be entirely sufficient to summarize
 *                the curve. After down-sampling, the curves will instead be made of approximately
 *                `numBins` points instead. Points are made from bins of equal numbers of
 *                consecutive points. The size of each bin is
 *                `floor(scoreAndLabels.count() / numBins)`, which means the resulting number
 *                of bins may not exactly equal numBins. The last bin in each partition may
 *                be smaller as a result, meaning there may be an extra sample at
 *                partition boundaries.
 */
class BinaryClassificationMetricsIMSPA (
    val scoreAndLabels: RDD[(Double, Double)],
    val numBins: Int) extends Logging {

  require(numBins >= 0, "numBins must be nonnegative")

  /**
   * Defaults `numBins` to 0.
   */
  def this(scoreAndLabels: RDD[(Double, Double)]) = this(scoreAndLabels, 0)

  /**
   * An auxiliary constructor taking a DataFrame.
   * @param scoreAndLabels a DataFrame with two double columns: score and label
   */
  private[mllib] def this(scoreAndLabels: DataFrame) =
    this(scoreAndLabels.map(r => (r.getDouble(0), r.getDouble(1))))

  /**
   * Unpersist intermediate RDDs used in the computation.
   */
  def unpersist() {
    cumulativeCounts.unpersist()
  }

  /**
   * Returns thresholds in descending order.
   */
  def thresholds(): RDD[Double] = cumulativeCounts.map(_._1)

  /**
   * Returns the receiver operating characteristic (ROC) curve,
   * which is an RDD of (false positive rate, true positive rate)
   * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
   * @see http://en.wikipedia.org/wiki/Receiver_operating_characteristic
   */
  def roc(): RDD[(Double, Double)] = {
    val rocCurve = createCurve(FalsePositiveRate, Recall)
    val sc = confusions.context
    val first = sc.makeRDD(Seq((0.0, 0.0)), 1)
    val last = sc.makeRDD(Seq((1.0, 1.0)), 1)
    new UnionRDD[(Double, Double)](sc, Seq(first, rocCurve, last))
  }

  /**
   * Computes the area under the receiver operating characteristic (ROC) curve.
   */
  def areaUnderROC(): Double = AreaUnderCurve.of(roc())

  /**
   * Returns the precision-recall curve, which is an RDD of (recall, precision),
   * NOT (precision, recall), with (0.0, 1.0) prepended to it.
   * @see http://en.wikipedia.org/wiki/Precision_and_recall
   */
  def pr(): RDD[(Double, Double)] = {
    val prCurve = createCurve(Recall, Precision)
    val sc = confusions.context
    val first = sc.makeRDD(Seq((0.0, 1.0)), 1)
    first.union(prCurve)
  }

  /**
   * Computes the area under the precision-recall curve.
   */
  def areaUnderPR(): Double = AreaUnderCurve.of(pr())

  /**
   * Returns the (threshold, F-Measure) curve.
   * @param beta the beta factor in F-Measure computation.
   * @return an RDD of (threshold, F-Measure) pairs.
   * @see http://en.wikipedia.org/wiki/F1_score
   */
  def fMeasureByThreshold(beta: Double): RDD[(Double, Double)] = createCurve(FMeasure(beta))

  /**
   * Returns the (threshold, F-Measure) curve with beta = 1.0.
   */
  def fMeasureByThreshold(): RDD[(Double, Double)] = fMeasureByThreshold(1.0)

  /**
   * Returns the (threshold, precision) curve.
   */
  def precisionByThreshold(): RDD[(Double, Double)] = createCurve(Precision)

  /**
   * Returns the (threshold, recall) curve.
   */
  def recallByThreshold(): RDD[(Double, Double)] = createCurve(Recall)

  def pGivenR(metricValue:Double): Double = getPrecisonByRecall(metricValue)

  private lazy val (
    cumulativeCounts: RDD[(Double, BinaryLabelCounter)],
    confusions: RDD[(Double, BinaryConfusionMatrix)]) = {
    // Create a bin for each distinct score value, count positives and negatives within each bin,
    // and then sort by score values in descending order.
    val counts = scoreAndLabels.combineByKey(
      createCombiner = (label: Double) => new BinaryLabelCounter(0L, 0L) += label,
      mergeValue = (c: BinaryLabelCounter, label: Double) => c += label,
      mergeCombiners = (c1: BinaryLabelCounter, c2: BinaryLabelCounter) => c1 += c2
    ).sortByKey(ascending = false)

    val binnedCounts =
      // Only down-sample if bins is > 0
      if (numBins == 0) {
        // Use original directly
        counts
      } else {
        val countsSize = counts.count()
        // Group the iterator into chunks of about countsSize / numBins points,
        // so that the resulting number of bins is about numBins
        var grouping = countsSize / numBins
        if (grouping < 2) {
          // numBins was more than half of the size; no real point in down-sampling to bins
          logInfo(s"Curve is too small ($countsSize) for $numBins bins to be useful")
          counts
        } else {
          if (grouping >= Int.MaxValue) {
            logWarning(
              s"Curve too large ($countsSize) for $numBins bins; capping at ${Int.MaxValue}")
            grouping = Int.MaxValue
          }
          counts.mapPartitions(_.grouped(grouping.toInt).map { pairs =>
            // The score of the combined point will be just the first one's score
            val firstScore = pairs.head._1
            // The point will contain all counts in this chunk
            val agg = new BinaryLabelCounter()
            pairs.foreach(pair => agg += pair._2)
            (firstScore, agg)
          })
        }
      }

    val agg = binnedCounts.values.mapPartitions { iter =>
      val agg = new BinaryLabelCounter()
      iter.foreach(agg += _)
      Iterator(agg)
    }.collect()
    val partitionwiseCumulativeCounts =
      agg.scanLeft(new BinaryLabelCounter())(
        (agg: BinaryLabelCounter, c: BinaryLabelCounter) => agg.clone() += c)
    val totalCount = partitionwiseCumulativeCounts.last
    logInfo(s"Total counts: $totalCount")
    val cumulativeCounts = binnedCounts.mapPartitionsWithIndex(
      (index: Int, iter: Iterator[(Double, BinaryLabelCounter)]) => {
        val cumCount = partitionwiseCumulativeCounts(index)
        iter.map { case (score, c) =>
          cumCount += c
          (score, cumCount.clone())
        }
      }, preservesPartitioning = true)
    cumulativeCounts.persist()
    val confusions = cumulativeCounts.map { case (score, cumCount) =>
      (score, BinaryConfusionMatrixImpl(cumCount, totalCount).asInstanceOf[BinaryConfusionMatrix])
    }
    (cumulativeCounts, confusions)
  }

  /** Creates a curve of (threshold, metric). */
  private def createCurve(y: BinaryClassificationMetricComputer): RDD[(Double, Double)] = {
    confusions.map { case (s, c) =>
      (s, y(c))
    }
  }

  /** Creates a curve of (metricX, metricY). */
  private def createCurve(
      x: BinaryClassificationMetricComputer,
      y: BinaryClassificationMetricComputer): RDD[(Double, Double)] = {
    confusions.map { case (_, c) =>
      (x(c), y(c))
    }
  }

  def getPrecisonByRecall(desired_recall: Double): Double = {

    var result = 0.0
    val precision = precisionByThreshold()
    val recall = recallByThreshold()
    val prByThreshold = precision.join(recall)

    val sc = SparkContext.getOrCreate()
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val prcurve = prByThreshold.toDF().select(col("_2._1").alias("precision"),col("_2._2").alias("recall"),col("_1").alias("threshold"))

    val prcurve_filtered = prcurve.where(col("recall").isin(desired_recall))
    if(prcurve_filtered.count() > 0)
    {
      result = prcurve_filtered.groupBy("recall").min("precision").take(1)(0)(1).toString.toDouble
    }

    else {
      val prcurve_with_recallDiff = prcurve.withColumn("diff", col("recall") - desired_recall)

      val greater_neighbour_df = prcurve_with_recallDiff.where(col("diff") > 0).sort(asc("diff")).select(col("recall"))
      val lesser_neighbour_df = prcurve_with_recallDiff.where(col("diff") < 0).sort(desc("diff")).select(col("recall"))
      greater_neighbour_df.cache()
      lesser_neighbour_df.cache()

      if (greater_neighbour_df.count() > 0 &&  lesser_neighbour_df.count() > 0) {

        val greater_neighbour_recall = (greater_neighbour_df.take(1)) (0)(0)
        val lesser_neighbour_recall = (lesser_neighbour_df.take(1)) (0)(0)

        val nn_precisions = prcurve_with_recallDiff.where(col("recall").isin(greater_neighbour_recall,lesser_neighbour_recall)).groupBy("recall").min("precision","diff").select("min(diff)","min(precision)").take(2)

        val diff_value_near1 = nn_precisions(0)(0).toString.toDouble.abs
        val diff_value_near2 = nn_precisions(1)(0).toString.toDouble.abs
        val precision_near1 = nn_precisions(0)(1).toString.toDouble
        val precision_near2 = nn_precisions(1)(1).toString.toDouble

        if (diff_value_near1 > diff_value_near2) {
          result = precision_near2
        }
        else if (diff_value_near1 < diff_value_near2) {
          result = precision_near1
        }
        else if (diff_value_near1 == diff_value_near2) {
          result = (precision_near1 + precision_near2) / 2.0
        }

      } else if(greater_neighbour_df.count() == 0 &&  lesser_neighbour_df.count() > 0) {

        val lesser_neighbour_recall = (lesser_neighbour_df.take(1)) (0)(0)
        val nn_precisions = prcurve_with_recallDiff.where(col("recall").isin(lesser_neighbour_recall)).groupBy("recall").min("precision","diff").select("min(diff)","min(precision)").take(1)
        result = nn_precisions(0)(1).toString.toDouble
      } else if(greater_neighbour_df.count() > 0 &&  lesser_neighbour_df.count() == 0) {

        val greater_neighbour_recall = (greater_neighbour_df.take(1)) (0)(0)
        val nn_precisions = prcurve_with_recallDiff.where(col("recall").isin(greater_neighbour_recall)).groupBy("recall").min("precision","diff").select("min(diff)","min(precision)").take(1)
        result = nn_precisions(0)(1).toString.toDouble
      }
    }
    result
  }
}
