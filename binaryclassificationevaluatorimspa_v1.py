from abc import abstractmethod, ABCMeta

import os
import time
import datetime
import random
import numpy as np
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
#from pyspark import since
from pyspark.ml.wrapper import JavaWrapper
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasRawPredictionCol
from pyspark.ml.util import keyword_only
from pyspark.mllib.common import inherit_doc
from pyspark.sql.functions import *

from crossvalidator import *
from stratification import *

__all__ = ['Evaluator', 'BinaryClassificationEvaluator_IMSPA']


partition_size = 20

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

def _Test1():
    # input parameters
    pos_file = "/dat_hae.csv"
    neg_file = "/dat_nonhae.csv"
    data_file = "/dat_results.csv"
    start_tree = 5
    stop_tree = 10
    num_tree = 2
    start_depth = 2
    stop_depth = 3
    num_depth = 2
    nFolds = 3

    metricName = "pGivenR"
    # set desired recall value for metric precisionByRecall
    metricValue = "0.8"

    s3_path = "s3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data"
    data_path = s3_path + ""
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = s3_path + "Results/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3)
    # resultDir_master = "~/Documents/DataFactZ/IMSUK/BackUp/datafactz/task_3/toydata/Results/" + st + "/"
    # if not os.path.exists(resultDir_master):
    #    os.makedirs(resultDir_master)

    # seed
    seed = 42
    random.seed(seed)

    # reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')
    # get the column names
    pos_col = pos.columns
    neg_col = neg.columns

    # combine features
    assembler_pos = VectorAssembler(inputCols=pos_col[2:], outputCol="features")
    assembler_neg = VectorAssembler(inputCols=neg_col[2:-1], outputCol="features")

    # get the input positive and negative dataframe
    pos_asmbl = assembler_pos.transform(pos) \
        .select('PATIENT_ID', 'HAE', 'features') \
        .withColumnRenamed('PATIENT_ID', 'matched_positive_id') \
        .withColumnRenamed('HAE', 'label')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    neg_asmbl = assembler_neg.transform(neg) \
        .select('HAE', 'HAE_PATIENT_ID', 'features') \
        .withColumnRenamed('HAE', 'label') \
        .withColumnRenamed('HAE_PATIENT_ID', 'matched_positive_id')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    data = pos_ori.unionAll(neg_ori)

    dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=nFolds)
    dataWithFoldID.cache()

    # iteration through all folds
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
        rf = RandomForestClassifier(labelCol="indexed", featuresCol="features")

        # Create the parameter grid builder
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, list(np.linspace(start_tree, stop_tree,
                                                   num_tree).astype('int'))) \
            .addGrid(rf.maxDepth, list(np.linspace(start_depth, stop_depth,
                                                   num_depth).astype('int'))) \
            .build()

        # Create the evaluator
        evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed", metricName=metricName, metricValue=metricValue)

        # Create the cross validator
        crossval = CrossValidatorWithStratification(estimator=rf,
                                                    estimatorParamMaps=paramGrid,
                                                    evaluator=evaluator,
                                                    numFolds=nFolds)

        # run cross-validation and choose the best parameters
        cvModel = crossval.fit(tr_td)

        # Predict on training data
        prediction_tr = cvModel.transform(tr_td)
        pred_score_tr = prediction_tr.select('label', 'indexed', 'probability')

        # predict on test data
        prediction_ts = cvModel.transform(ts_td)
        pred_score_ts = prediction_ts.select('label', 'indexed', 'probability')

        # AUC
        #prediction_tr.show(truncate=False)
        AUC_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'areaUnderROC'})
        AUC_ts = evaluator.evaluate(prediction_ts, {evaluator.metricName: 'areaUnderROC'})

        PR_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'areaUnderPR'})
        pGivenR_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'pGivenR'})

        #print(AUC_tr)
        #print(AUC_ts)
        #print(PR_tr)
        #print(pGivenR_tr)

        # print out results
        # fAUC = open(resultDir_master + "AUC_fold" + str(iFold) + ".txt", "a")
        # fAUC.write("{}: Traing AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_tr))
        # fAUC.write("{}: Test AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_ts))
        # fAUC.close()

        pred_score_tr.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_tr_fold" + str(iFold) + ".csv")
        pred_score_ts.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_ts_fold" + str(iFold) + ".csv")

        # fFinished = open(resultDir_master + "finished.txt", "a")
        # fFinished.write("Test for {} finished. Please manually check the result.. \n".format(data_file))
        # fFinished.close()


def _Test2():
    # input parameters
    pos_file = "/dat_hae.csv"
    neg_file = "/dat_nonhae.csv"
    data_file = "/dat_results.csv"
    start_tree = 5
    stop_tree = 10
    num_tree = 2
    start_depth = 2
    stop_depth = 3
    num_depth = 2
    nFolds = 3

    s3_path = "s3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data"
    data_path = s3_path + ""
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
    resultDir_s3 = s3_path + "Results/" + st + "/"
    if not os.path.exists(resultDir_s3):
        os.makedirs(resultDir_s3)
    # resultDir_master = "~/Documents/DataFactZ/IMSUK/BackUp/datafactz/task_3/toydata/Results/" + st + "/"
    # if not os.path.exists(resultDir_master):
    #    os.makedirs(resultDir_master)

    # seed
    seed = 42
    random.seed(seed)

    # reading in the data from S3
    pos = sqlContext.read.load((data_path + pos_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')

    neg = sqlContext.read.load((data_path + neg_file),
                               format='com.databricks.spark.csv',
                               header='true',
                               inferSchema='true')
    # get the column names
    pos_col = pos.columns
    neg_col = neg.columns

    # combine features
    assembler_pos = VectorAssembler(inputCols=pos_col[2:], outputCol="features")
    assembler_neg = VectorAssembler(inputCols=neg_col[2:-1], outputCol="features")

    # get the input positive and negative dataframe
    pos_asmbl = assembler_pos.transform(pos) \
        .select('PATIENT_ID', 'HAE', 'features') \
        .withColumnRenamed('PATIENT_ID', 'matched_positive_id') \
        .withColumnRenamed('HAE', 'label')

    pos_ori = pos_asmbl.withColumn('label', pos_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    neg_asmbl = assembler_neg.transform(neg) \
        .select('HAE', 'HAE_PATIENT_ID', 'features') \
        .withColumnRenamed('HAE', 'label') \
        .withColumnRenamed('HAE_PATIENT_ID', 'matched_positive_id')

    neg_ori = neg_asmbl.withColumn('label', neg_asmbl['label'].cast('double')) \
        .select('matched_positive_id', 'label', 'features')

    data = pos_ori.unionAll(neg_ori)

    dataWithFoldID = AppendDataMatchingFoldIDs(data=data, nFolds=nFolds)
    dataWithFoldID.cache()

    # iteration through all folds
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
        rf = RandomForestClassifier(labelCol="indexed", featuresCol="features")

        # Create the parameter grid builder
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, list(np.linspace(start_tree, stop_tree,
                                                   num_tree).astype('int'))) \
            .addGrid(rf.maxDepth, list(np.linspace(start_depth, stop_depth,
                                                   num_depth).astype('int'))) \
            .build()

        # Create the evaluator
        evaluator = BinaryClassificationEvaluator_IMSPA(labelCol="indexed")
        #evaluator = BinaryClassificationEvaluator()

        # Create the cross validator
        crossval = CrossValidatorWithStratification(estimator=rf,
                                                    estimatorParamMaps=paramGrid,
                                                    evaluator=evaluator,
                                                    numFolds=nFolds)

        # run cross-validation and choose the best parameters
        cvModel = crossval.fit(tr_td)

        # Predict on training data
        prediction_tr = cvModel.transform(tr_td)
        pred_score_tr = prediction_tr.select('label', 'indexed', 'probability')

        # predict on test data
        prediction_ts = cvModel.transform(ts_td)
        pred_score_ts = prediction_ts.select('label', 'indexed', 'probability')

        # AUC
        #prediction_tr.show(truncate=False)
        AUC_tr = evaluator.evaluate(prediction_tr, {evaluator.metricName: 'areaUnderROC'})
        AUC_ts = evaluator.evaluate(prediction_ts, {evaluator.metricName: 'areaUnderROC'})

        # print out results
        # fAUC = open(resultDir_master + "AUC_fold" + str(iFold) + ".txt", "a")
        # fAUC.write("{}: Traing AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_tr))
        # fAUC.write("{}: Test AUC = {} \n".format(data_file[:(len(data_file) - 4)], AUC_ts))
        # fAUC.close()

        pred_score_tr.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_tr_fold" + str(iFold) + ".csv")
        pred_score_ts.coalesce(1) \
            .write.format('com.databricks.spark.csv') \
            .save(resultDir_s3 + data_file[:(len(data_file) - 4)] + "_pred_ts_fold" + str(iFold) + ".csv")

        # fFinished = open(resultDir_master + "finished.txt", "a")
        # fFinished.write("Test for {} finished. Please manually check the result.. \n".format(data_file))
        # fFinished.close()

def _Test3():
    path = "s3://emr-rwes-pa-spark-dev-datastore/Hui/datafactz_test/01_data/task10/labelPred.csv"
    scoreAndLabels = sqlContext.read.load(path, format='com.databricks.spark.csv', header='true',
                                              inferSchema='true')
    scoreAndLabels = scoreAndLabels.select(scoreAndLabels.label.cast('double').alias('label'), scoreAndLabels.pred)
    scoreAndLabelsRaw = avg_asmb(scoreAndLabels)
    evaluator = BinaryClassificationEvaluator_IMSPA()
    desiredRecall = "0.4"

    precision = evaluator.evaluate(scoreAndLabelsRaw,
                                   {evaluator.metricName: "pGivenR",
                                    evaluator.metricValue: desiredRecall})

    print("Precision to given recall is %s" % precision)


if __name__ == "__main__":
    from pyspark.context import SparkContext
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.sql.types import *

    app_name = "BinaryClassificationEvaluator_IMSPA"
    sc = SparkContext(appName=app_name)
    sqlContext = SQLContext(sc)

    print('Test1 - Check pGivenR')
    _Test1()

    print('Test2 - Check ROC PR')
    _Test2()

    print('Test3 - Check precison value for recall 0.4: ( Should be around 0.9048 )')
    _Test3()
