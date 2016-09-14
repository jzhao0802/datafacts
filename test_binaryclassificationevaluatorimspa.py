from abc import abstractmethod, ABCMeta

import os
import time
import datetime
import random
import numpy as np
from pyspark.ml import Estimator
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark import since
from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasRawPredictionCol
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql import functions as F
import unittest
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import decimal
from crossvalidator import *
from stratification import *

__all__ = ['Evaluator', 'BinaryClassificationEvaluator_IMSPA']

app_name = "Test_BinaryClassificationEvaluator_IMSPA"
sc = SparkContext(appName=app_name)
sqlContext = SQLContext(sc)
nfolds = 5

def avg_asmb(data):
    newdata = data \
        .withColumn('avg_prob_0', lit(1) - data.pred) \
        .withColumnRenamed('pred', 'avg_prob_1')
    asmbl = VectorAssembler(inputCols=['avg_prob_0', 'avg_prob_1'],
                            outputCol="rawPrediction")
    # get the input positive and negative dataframe
    data_asmbl = asmbl.transform(newdata) \
        .select('label', 'rawPrediction')
    return data_asmbl

path = 's3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data/task10/labelPred.csv'
scoreAndLabels = sqlContext.read.load(path, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double').alias('label'), scoreAndLabels.pred)
scoreAndLabelsRaw = avg_asmb(scoreAndLabels)
scoreAndLabelsRaw.cache()

path2 = 's3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data/task10/randata_100x20.csv'
scoreAndLabels2 = sqlContext.read.load(path2, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
scoreAndLabelscol = scoreAndLabels2.columns
# combine features
assembler_scoreAndLabels = VectorAssembler(inputCols=scoreAndLabelscol[2:], outputCol="features")
data = assembler_scoreAndLabels.transform(scoreAndLabels2) \
            .select('matched_positive_id', 'label', 'features')
data = data.select(data.matched_positive_id, data.label.cast('double').alias('label'), data.features)
dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds= nfolds)
dataWithFoldID.cache()

@inherit_doc
class Evaluator(Params):
    """
    Base class for evaluators that compute metrics from predictions.

    .. versionadded:: 1.4.0
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _evaluate(self, dataset):
        """
        Evaluates the output.

        :param dataset: a dataset that contains labels/observations and
               predictions
        :return: metric
        """
        raise NotImplementedError()

    #@since("1.4.0")
    def evaluate(self, dataset, params=None):
        """
        Evaluates the output with optional parameters.

        :param dataset: a dataset that contains labels/observations and
                        predictions
        :param params: an optional param map that overrides embedded
                       params
        :return: metric
        """
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._evaluate(dataset)
            else:
                return self._evaluate(dataset)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))

    #@since("1.5.0")
    def isLargerBetter(self):
        """
        Indicates whether the metric returned by :py:meth:`evaluate` should be maximized
        (True, default) or minimized (False).
        A given evaluator may support multiple metrics which may be maximized or minimized.
        """
        return True


@inherit_doc
class JavaEvaluator(Evaluator, JavaWrapper):
    """
    Base class for :py:class:`Evaluator`s that wrap Java/Scala
    implementations.
    """

    __metaclass__ = ABCMeta

    def _evaluate(self, dataset):
        """
        Evaluates the output.
        :param dataset: a dataset that contains labels/observations and predictions.
        :return: evaluation metric
        """
        self._transfer_params_to_java()
        return self._java_obj.evaluate(dataset._jdf)

    def isLargerBetter(self):
        self._transfer_params_to_java()
        return self._java_obj.isLargerBetter()


@inherit_doc
class BinaryClassificationEvaluator_IMSPA(JavaEvaluator, HasLabelCol, HasRawPredictionCol):
    """
    Evaluator for binary classification, which expects two input
    columns: rawPrediction and label.

    >>> from pyspark.mllib.linalg import Vectors
    >>> scoreAndLabels = map(lambda x: (Vectors.dense([1.0 - x[0], x[0]]), x[1]),
    ...    [(0.1, 0.0), (0.1, 1.0), (0.4, 0.0), (0.6, 0.0), (0.6, 1.0), (0.6, 1.0), (0.8, 1.0)])
    >>> dataset = sqlContext.createDataFrame(scoreAndLabels, ["raw", "label"])
    ...
    >>> evaluator = BinaryClassificationEvaluator(rawPredictionCol="raw")
    >>> evaluator.evaluate(dataset)
    0.70...
    >>> evaluator.evaluate(dataset, {evaluator.metricName: "areaUnderPR"})
    0.83...

    .. versionadded:: 1.4.0
    """

    # a placeholder to make it appear in the generated doc
    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation (areaUnderROC|areaUnderPR|pGivenR)")

    @keyword_only
    def __init__(self, rawPredictionCol="rawPrediction", labelCol="label",
                 metricName="areaUnderROC", metricValue="0.5"):
        """
        __init__(self, rawPredictionCol="rawPrediction", labelCol="label", \
                 metricName="areaUnderROC", metricValue=0.5)
        """
        super(BinaryClassificationEvaluator_IMSPA, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.imshealth.ml.evaluation.BinaryClassificationEvaluatorIMSPA", self.uid)

        #: param for metric name in evaluation (areaUnderROC|areaUnderPR|pGivenR)
        self.metricName = Param(self, "metricName",
                                "metric name in evaluation (areaUnderROC|areaUnderPR|pGivenR)")
        self.metricValue = Param(self, "metricValue", "metric recall value in pGivenR")
        self._setDefault(rawPredictionCol="rawPrediction", labelCol="label",
                         metricName="areaUnderROC", metricValue="0.5")
        kwargs = self.__init__._input_kwargs
        self._set(**kwargs)

    #@since("1.4.0")
    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        self._paramMap[self.metricName] = value
        return self

    #@since("1.4.0")
    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)

    #@since("1.4.0")
    def setMetricValue(self, value):
        """
        Sets the value of :py:attr:`metricValue`.
        """
        self._paramMap[self.metricValue] = value
        return self

    #@since("1.4.0")
    def getMetricValue(self):
        """
        Gets the value of metricValue or its default value.
        """
        return self.getOrDefault(self.metricValue)

    # @since("1.4.0")
    @keyword_only
    def setParams(self, rawPredictionCol="rawPrediction", labelCol="label",
                  metricName="areaUnderROC", metricValue="0.5"):
        """
        setParams(self, rawPredictionCol="rawPrediction", labelCol="label", \
                  metricName="areaUnderROC", metricValue="0.5")
        Sets params for binary classification evaluator.
        """
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

class PREvaluationMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nFolds = 5
        #app_name = "BinaryClassificationEvaluator_IMSPA"
        #sc = SparkContext(appName=app_name)
        #sqlContext = SQLContext(sc)
        
		
        

    def test_is_ROC_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        ROC = evaluator.evaluate(scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderROC'})
        tolerance = 0.0050
        self.assertTrue((0.8290 - tolerance) <= ROC<= (0.8290 + tolerance), "ROC value is outside of the specified range")

    def test_is_ROC_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        ROC = evaluator.evaluate(scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderROC'})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_PR_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        tolerance = 0.0050
        PR = evaluator.evaluate(scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue((0.8372 - tolerance) <= PR <= (0.8372 + tolerance), "PR value is outside of the specified range")

    def test_is_PR_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        PR = evaluator.evaluate(scoreAndLabelsRaw, {evaluator.metricName: 'areaUnderPR'})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")

    def test_is_precision_matching_1(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = '0.2'
        precision = evaluator.evaluate(scoreAndLabelsRaw,
                                       {evaluator.metricName: "pGivenR",
                                        evaluator.metricValue: desiredRecall})
        self.assertAlmostEqual(precision, 1.0, places=3, msg="precisionByRecall metric producing incorrect precision: %s" % precision)

    def test_is_precision_matching_2(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = '0.4'
        precision = evaluator.evaluate(scoreAndLabelsRaw,
                                       {evaluator.metricName: "pGivenR",
                                        evaluator.metricValue: desiredRecall})
        self.assertAlmostEqual(precision, 0.9048, places=3, msg="precisionByRecall metric producing incorrect precision: %s" % precision)
        print("%s \n" % precision)

    def test_is_precision_matching_3(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = '0.6'
        precision = evaluator.evaluate(scoreAndLabelsRaw,
                                       {evaluator.metricName: "pGivenR",
                                        evaluator.metricValue: desiredRecall})
        self.assertAlmostEqual(precision, 0.8003, places=3, msg="precisionByRecall metric producing incorrect precision: %s" % precision)
        print("%s \n" % precision)

    def test_is_precision_isLargeBetter_matching(self):
        evaluator = BinaryClassificationEvaluator_IMSPA()
        desiredRecall = '0.2'
        precision = evaluator.evaluate(scoreAndLabelsRaw,
                                       {evaluator.metricName: "pGivenR",
                                        evaluator.metricValue: desiredRecall})
        self.assertTrue(evaluator.isLargerBetter(), "method isLargerBetter() returning false.")


    def test_is_best_metric_correct(self):
        nFolds = nfolds
        lambdastart = 0.0001
        lambdastop = 0.001
        lambdanum = 2
        for iFold in range(nFolds):
            # stratified sampling
            ts = dataWithFoldID.filter(dataWithFoldID.foldID == iFold)
            tr = dataWithFoldID.filter(dataWithFoldID.foldID != iFold)

            # remove the fold id column
            ts = ts.drop('foldID')
            tr = tr.drop('foldID')

            # transfer to RF invalid label column
            stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
            si_model = stringIndexer.fit(tr)
            tr_td = si_model.transform(tr)
            ts_td = si_model.transform(ts)

            # Build the model
            """
            Set the ElasticNet mixing parameter.
            For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
            For 0 < alpha < 1, the penalty is a combination of L1 and L2.
            Default is 0.0 which is an L2 penalty.
            """
            lr = LogisticRegression(featuresCol="features",
                                    labelCol="label",
                                    fitIntercept=True,
                                    elasticNetParam=0.0)

            # Create the parameter grid builder
            paramGrid = ParamGridBuilder() \
                .addGrid(lr.regParam, list(np.linspace(lambdastart, lambdastop, lambdanum))) \
                .build()

            # Create the evaluator
            evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed", metricName="pGivenR")

            # Create the cross validator
            crossval = CrossValidatorWithStratification(estimator=lr,
                                                        estimatorParamMaps=paramGrid,
                                                        evaluator=evaluator,
                                                        numFolds=nFolds)

            # run cross-validation and choose the best parameters
            cvModel = crossval.fit(tr_td)

            self.assertEqual(crossval.getBestMetric(), (crossval.getAllMetrics()).max(),"best metric does not correspond to the maximum.")

if __name__ == "__main__":
    unittest.main()
