// Databricks notebook source
// AutoML imports 
/* import com.databricks.labs.automl.executor.config.ConfigurationGenerator
import com.databricks.labs.automl.exploration.FeatureImportances
import com.databricks.labs.automl.executor.FamilyRunner
import com.databricks.labs.automl.pipeline.inference.PipelineModelInference */

// spark sql imports 
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.functions.translate
import org.apache.spark.sql.functions.log

// spark ml imports 
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder

// xgboost import
import ml.dmlc.xgboost4j.scala.spark._

// COMMAND ----------

// MAGIC %md
// MAGIC # DBFS  
// MAGIC The default storage location in DBFS is the DBFS root. Within DBFS, there are several root locations. For this case, we access the data from `/FileStore`, where imported data files and libraries are stored by default. 

// COMMAND ----------

// DO NOT RUN

// delete the csv file (for testing upload configurations)
var result = dbutils.fs.rm("dbfs:/FileStore/shared_uploads/skim658@gwu.edu/listings.csv")
print(result)
// fs.rm returns a boolean, but just double checking that the file was removed 
dbutils.fs.ls("dbfs:/FileStore/shared_uploads/skim658@gwu.edu/listings.csv")

// COMMAND ----------

// data file path within the dbfs filestore  
val filePath = ("dbfs:/FileStore/shared_uploads/skim658@gwu.edu/listings.csv")

// read in the data 
val rawData = spark.read
  .option("header", "true")
  .option("multiLine", "true")
  .option("inferSchema", "true")
  .csv(filePath)

// COMMAND ----------

// MAGIC %md
// MAGIC # Initial Exploratory Analysis

// COMMAND ----------

// peak at the raw data (before any preprocessing)
display(rawData)

// COMMAND ----------

// check the columns 
rawData.columns

// COMMAND ----------

// check data types 
rawData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC # Feature Engineering (SECTION NOT USED)

// COMMAND ----------

val labelColumn = "price"

// COMMAND ----------

// generic feature overrides 
val feOverrides = Map(
  "labelCol" -> labelColumn,
  "naFillFlag" -> true,
  "varianceFilterFlag" -> true,
  "tunerTrainSplitMethod" -> "stratified",
  "autoStoppingFlag" -> true,
  "tunerAutoStoppingScore" -> 0.90
)

// get feature importances (using XGboost)
val fiConfig = ConfigurationGenerator.generateConfigFromMap("XGBoost", "regressor", feOverrides)
val fiMainConfig = ConfigurationGenerator.generateFeatureImportanceConfig(fiConfig)
val importances = new FeatureImportances(rawData, fiMainConfig, "count", 1).generateFeatureImportances()

// COMMAND ----------

display(importances.importances)

// COMMAND ----------

// MAGIC %md
// MAGIC # Initial Data Cleaning

// COMMAND ----------

// MAGIC %md
// MAGIC Chosen data columns:  
// MAGIC 
// MAGIC | Data Column                         | Type            | Description                                                               |
// MAGIC |-------------------------------------|-----------------|---------------------------------------------------------------------------|
// MAGIC | host_is_superhost                   | boolean [1]     |                                                                           |
// MAGIC | host_total_listings_count           | string          | Number of listings the host has per Airbnb calculations.                  |
// MAGIC | host_identity_verified              | boolean [1]     |                                                                           |
// MAGIC | latitude                            | double          | Uses the World Geodetic System (WGS84) projection,                        |
// MAGIC | longitude                           | double [1]      | Uses the World Geodetic System (WGS84) projection                         |
// MAGIC | room_type                           | string          | Entire home/apt, private room, shared room, hotel.                        |
// MAGIC | accommodates                        | integer [1]     | Maximum capacity of the listing.                                          |
// MAGIC | bathrooms_text                      | string          | The number of bathrooms in the listing. On the Airbnb web-site, the bathrooms field has evolved from a number to a textual description. For older scrapes, bathrooms is used.                         |
// MAGIC | bedrooms                            | integer         | The number of bedrooms.                                                   |
// MAGIC | beds                                | integer         | The number of bed(s).                                                     |
// MAGIC | **price** (target label)            | double [1]      | Daily price in local currency.                                            |
// MAGIC | minimum_nights                      | integer         | Minimum number of night stay for the listing.                             |
// MAGIC | number_of_reviews                   | double          | The number of reviews the listing has.                                    |
// MAGIC | review_scores_rating                | double          | Review score average.                                                     |
// MAGIC | review_scores_accuracy              | double          | Review accuracy score average.                                            |
// MAGIC | review_scores_cleanliness           | double          | Review cleanliness score average.                                         |
// MAGIC | review_scores_checkin               | double [1]      | Review checkin score average.                                             |
// MAGIC | review_scores_communication         | double          | Review communication score average.                                       |
// MAGIC | review_scores_location              | double          | Review location score average.                                            |
// MAGIC | review_scores_value                 | double          | Review value score average.                                               |
// MAGIC | instant_bookable                    | boolean [1]     | Whether the guest can automatically book the listing without the host requiring to accept their booking request. An indicator of a commercial listing.                                            |
// MAGIC | reviews_per_month                   | double          | The number of reviews the listing has over the lifetime of the listing per month.                                            |
// MAGIC 
// MAGIC [1]: inferred as a string when reading in data; will need to fix 
// MAGIC  - for booleans: t = true; f = false
// MAGIC  - for other data types most likely due to null records

// COMMAND ----------

// select the columns that we will use for modeling
var selectedData = rawData.select(
  "host_is_superhost",
  "host_total_listings_count",
  "host_identity_verified",
  "latitude",
  "longitude",
  "room_type",
  "accommodates",
  "bathrooms_text",
  "bedrooms",
  "beds",
  "price",
  "minimum_nights",
  "number_of_reviews",
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value",
  "instant_bookable",
  "reviews_per_month")

// cache the selected data
selectedData.cache().count
display(selectedData)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we need to clean up the data types. As seen above, many fields were inferred as strings when they should be of other data types. Eventually, we need convert all data fields to numeric types for model inputs. 

// COMMAND ----------

// check schema for which data columns will have to be converted from categorical or textual to numerical
selectedData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Cast Target Column (Price) as Double  
// MAGIC We need to first remove the $'s

// COMMAND ----------

// fixing price column, removing the '$' symbol and any commas, then casting the values as a double 
selectedData = selectedData.withColumn("price", translate($"price", "$,", "").cast("double"))
display(selectedData)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Cast Integer Columns to Double

// COMMAND ----------

// columns that are of type integer 
val intCols = for (x <- selectedData.schema.fields if (x.dataType == IntegerType)) yield x.name

for (c <- intCols)
  selectedData = selectedData.withColumn(c, col(c).cast("double"))

// print out which columns were converted 
val columns = intCols.mkString("\n - ")
println(s"Columns converted from Int to Double:\n - $columns \n")

// COMMAND ----------

display(selectedData)

// COMMAND ----------

selectedData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC #### Check the DF Summary

// COMMAND ----------

display(selectedData.summary())

// COMMAND ----------

// MAGIC %md
// MAGIC #### Handle Bathrooms Text 
// MAGIC Going to isolate the number in the `bathrooms_text` column

// COMMAND ----------

// add new column called bathrooms resulting from splitting the bathrooms_text string and taking the first index; then cast it as a double 
selectedData = selectedData.withColumn("bathrooms", split($"bathrooms_text", " ")(0).cast("double"))
display(selectedData)

// COMMAND ----------

// check that bathrooms is of type double 
selectedData.printSchema()

// COMMAND ----------

// drop bathrooms_text column 
selectedData = selectedData.drop("bathrooms_text")
display(selectedData)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Handle Null and Invalid Values  
// MAGIC Using SparkML's Imputer (SparkML Imputer doesn't support categorical features)

// COMMAND ----------

// see number of null values 
display(selectedData.select(selectedData.columns.map(c => count(when(col(c).isNull || col(c) === "", c)).alias(c)): _*))

// COMMAND ----------

// MAGIC %md
// MAGIC The numerical column null values will be imputed using SparkML's. Since there are only a few rows with null or invalid values for the categorical columns, we will just drop them before one hot encoding in the next section.  
// MAGIC 
// MAGIC Check the following categorical columns for null/invalid values:  
// MAGIC - `host_is_superhost`
// MAGIC - `host_identity_verified`

// COMMAND ----------

// check the null row in the host_is_superhost column 
display(selectedData.filter($"host_is_superhost".isNull))

// COMMAND ----------

// check the invalid row in the host_is_superhost column 
display(selectedData.filter($"host_is_superhost" =!= "t" && $"host_is_superhost" =!= "f"))

// COMMAND ----------

// drop these two rows
selectedData = selectedData.filter(($"host_is_superhost" === "t" || $"host_is_superhost" === "f") && !($"host_is_superhost".isNull))
selectedData.count

// COMMAND ----------

// MAGIC %md
// MAGIC Check the `host_identity_verified` invalid were removed when we removed the invalid rows for `host_is_superhost`.

// COMMAND ----------

// check the null row in the host_identity_verified column 
display(selectedData.filter($"host_identity_verified".isNull))

// COMMAND ----------

// check the invalid row in the host_identity_verified column  
display(selectedData.filter($"host_identity_verified" =!= "t" && $"host_identity_verified" =!= "f"))

// COMMAND ----------

// MAGIC %md
// MAGIC Recheck null values. 

// COMMAND ----------

// recheck null values
display(selectedData.select(selectedData.columns.map(c => count(when(col(c).isNull || col(c) === "", c)).alias(c)): _*))

// COMMAND ----------

// MAGIC %md
// MAGIC When you impute for null values, have to add an addiitonal field specifying that the field was imputed. 

// COMMAND ----------

val imputeCols = Array(
  "bedrooms",
  "bathrooms",
  "beds",
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_value",
  "review_scores_location",
  "reviews_per_month"
)

for (c <- imputeCols) {
  selectedData = selectedData.withColumn(c + "_na", when(col(c).isNull, 1.0).otherwise(0.0))
  println(c)
}

// COMMAND ----------

// see new columns indicating imputation techniques were used on that column 
display(selectedData)

// COMMAND ----------

// MAGIC %md
// MAGIC SparkML's imputer requires all columns to be imputed are of type double. Do one last check to make sure all columns are of type double.

// COMMAND ----------

selectedData.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC Have to cast `review_scores_checkin` as double. 

// COMMAND ----------

selectedData = selectedData.withColumn("review_scores_checkin", $"review_scores_checkin".cast("double"))
selectedData.printSchema()

// COMMAND ----------

val imputer = new Imputer()
  .setStrategy("median")
  .setInputCols(imputeCols)
  .setOutputCols(imputeCols)

val imputedDf = imputer.fit(selectedData).transform(selectedData)

// COMMAND ----------

// recheck null values
display(imputedDf.select(imputedDf.columns.map(c => count(when(col(c).isNull || col(c) === "", c)).alias(c)): _*))

// COMMAND ----------

display(imputedDf)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Check for Potentially Extreme Values 

// COMMAND ----------

display(imputedDf.select("price").describe())

// COMMAND ----------

// MAGIC %md
// MAGIC The minimum value for the price column is zero. Let's see how many listings have a 0 for price. 

// COMMAND ----------

imputedDf.filter($"price" === 0).count

// COMMAND ----------

// MAGIC %md
// MAGIC We're only going to keep listings that have a positive (greater than 0) price. 

// COMMAND ----------

val positiveDf = imputedDf.filter($"price" > 0)
positiveDf.filter($"price" === 0).count

// COMMAND ----------

// MAGIC %md
// MAGIC Now let's check the minimum night values. 

// COMMAND ----------

display(positiveDf.select("minimum_nights").describe())

// COMMAND ----------

display(positiveDf.groupBy("minimum_nights").count().orderBy($"count".desc, $"minimum_nights"))

// COMMAND ----------

// MAGIC %md
// MAGIC A minimum night stay requirement of 600 and 1,125 nights seems excessive. Let's cap it off at a maximum value of 365 for one year.

// COMMAND ----------

val minNightDf = positiveDf.filter($"minimum_nights" <= 365)
display(minNightDf.groupBy("minimum_nights").count().orderBy($"count".desc, $"minimum_nights"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Save Dataframe

// COMMAND ----------

val filepath = "dbfs:/user/skim658@gwu.edu/dc_airbnb/dc_cleansed.parquet"
minNightDf.write.mode("overwrite").parquet(filepath)

// COMMAND ----------

// check the save
display(dbutils.fs.ls("dbfs:/user/skim658@gwu.edu/dc_airbnb"))

// COMMAND ----------

// MAGIC %md
// MAGIC # Load in Cleansed Data

// COMMAND ----------

// read in cleansed dataframe 
val inputPath = "dbfs:/user/skim658@gwu.edu/dc_airbnb/dc_cleansed.parquet"
val airbnbDF = spark.read.parquet(inputPath)
display(airbnbDF)

// COMMAND ----------

// MAGIC %md
// MAGIC # Linear Regression

// COMMAND ----------

// MAGIC %md
// MAGIC #### Train/Test Split
// MAGIC Using a seed for reproducibility. 

// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(0.8, 0.2), seed = 42)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Handle Categorical Variables
// MAGIC Going to use One Hot Encoding (or "dummy" variables).

// COMMAND ----------

trainDF.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC We first need to use a `StringIndexer`, then we can apply the `OneHotEncoder` to the output of the `StringIndexer`. 

// COMMAND ----------

val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")
val oheOutputCols = categoricalCols.map(_ + "OHE")

val stringIndexer = new StringIndexer()
  .setInputCols(categoricalCols)
  .setOutputCols(indexOutputCols)
  .setHandleInvalid("skip")

val oheEncoder = new OneHotEncoder()
  .setInputCols(indexOutputCols)
  .setOutputCols(oheOutputCols)

// COMMAND ----------

// MAGIC %md
// MAGIC Now need to combine our OHE features back with our numeric features and use a `VectorAssembler`. Spark models (such as Linear Regression) expect a column of Vector type as input. 

// COMMAND ----------

val numericCols = trainDF.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = oheOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
  .setInputCols(assemblerInputs)
  .setOutputCol("features")

// COMMAND ----------

val lin_reg = new LinearRegression().setLabelCol("price").setFeaturesCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Create the Pipeline
// MAGIC We can put all these transformations into a `Pipeline`. That way we don't have to remember all the transformations or the ordering of the transformations in the future. Instead of eventually saving the model, we can save the pipeline and save some time and complexity having everything in one construct/spot.

// COMMAND ----------

val lrStages = Array(stringIndexer, oheEncoder, vecAssembler, lin_reg)
val lrPipeline = new Pipeline().setStages(lrStages)
val lrPipelineModel = lrPipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %md
// MAGIC Save the fitted pipeline for future use. 

// COMMAND ----------

val lrPipelinePath = "dbfs:/user/skim658@gwu.edu/dc_airbnb/lr_pipeline_model"
lrPipelineModel.write.overwrite().save(lrPipelinePath)

// COMMAND ----------

// check the save
display(dbutils.fs.ls("dbfs:/user/skim658@gwu.edu/dc_airbnb"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Run Model on Test Set 

// COMMAND ----------

val predDF = lrPipelineModel.transform(testDF)
display(predDF.select("features", "price", "prediction"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Evaluate the Model

// COMMAND ----------

val lrEvaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("price").setMetricName("rmse")
val lrRmse = lrEvaluator.evaluate(predDF)
val lrR2 = lrEvaluator.setMetricName("r2").evaluate(predDF)
println(s"RMSE: $lrRmse")
println(s"R2: $lrR2")
println("-" * 100)

// COMMAND ----------

// MAGIC %md
// MAGIC # XGBoost  
// MAGIC One hot encoding is not needed for XGBoost. We will just use the `StringIndexer` and the `VectorAssembler`. Since we have a large range of potential prices, we will log transform the target label, `log(price)`, in order to reduce the relative importance of the extreme values and generalize better. We are accepting a potentially larger error predicting extreme values but will hopefully generalize on the bulk of the data better. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Train/Test Split

// COMMAND ----------

val Array(trainDF2, testDF2) = airbnbDF.withColumn("label", log($"price")).randomSplit(Array(0.8, 0.2), seed = 42)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Handle Categorical Variables

// COMMAND ----------

val categoricalCols2 = trainDF2.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols2 = categoricalCols2.map(_ + "Index")

val stringIndexer2 = new StringIndexer()
  .setInputCols(categoricalCols2)
  .setOutputCols(indexOutputCols2)
  .setHandleInvalid("skip")

val numericCols2 = trainDF2.dtypes.filter{ case (field, dataType) => dataType == "DoubleType" && field != "price" && field != "label"}.map(_._1)
val assemblerInputs2 = indexOutputCols2 ++ numericCols2
val vecAssembler2 = new VectorAssembler()
  .setInputCols(assemblerInputs2)
  .setOutputCol("features")

val xgPipeline = new Pipeline()
  .setStages(Array(stringIndexer2, vecAssembler2))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Define and Train the Model

// COMMAND ----------

val paramMap = List("num_round" -> 100, "eta" -> 0.1, "max_leaf_nodes" -> 50, "seed" -> 42, "missing" -> 0).toMap

val xgboostEstimator = new XGBoostRegressor(paramMap)

val xgboostPipeline = new Pipeline().setStages(xgPipeline.getStages ++ Array(xgboostEstimator))

val xgboostPipelineModel = xgboostPipeline.fit(trainDF2)

// COMMAND ----------

// MAGIC %md
// MAGIC Save the fitted pipeline for future use.

// COMMAND ----------

val xgPipelinePath = "dbfs:/user/skim658@gwu.edu/dc_airbnb/xg_pipeline_model"
xgboostPipeline.write.overwrite().save(xgPipelinePath)

// COMMAND ----------

// check the save
display(dbutils.fs.ls("dbfs:/user/skim658@gwu.edu/dc_airbnb"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Run Model on Test Set

// COMMAND ----------

val xgboostLogPredictedDF = xgboostPipelineModel.transform(testDF2)

val expXgboostDF = xgboostLogPredictedDF.withColumn("prediction", exp(col("prediction")))

// COMMAND ----------

display(expXgboostDF.select("price", "prediction"))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Evaluate the Model

// COMMAND ----------

val regressionEvaluator2 = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val rmse = regressionEvaluator2.evaluate(expXgboostDF)
val r2 = regressionEvaluator2.setMetricName("r2").evaluate(expXgboostDF)
println(s"RMSE: $rmse")
println(s"R2: $r2")
println("-"*100)
