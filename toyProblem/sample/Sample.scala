package com.imshealth

import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object Sample{
    def computeSum(dataset: DataFrame): Long = {
        dataset.agg(sum("b")).first.getLong(0)
    } 
} 
