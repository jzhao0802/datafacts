from pyspark import SparkContext
from pyspark.sql import SQLContext

app_name = "sample"
sc = SparkContext(appName = app_name)
sqlContext = SQLContext(sc)

df = sqlContext.createDataFrame([(1,2),(2,4),(3,8)],['a','b'])

sampleobj = sc._jvm.com.imshealth.Sample
sumValue = sampleobj.computeSum(df._jdf)

print(sumValue)
