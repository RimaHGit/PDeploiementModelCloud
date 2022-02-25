from pyspark.sql import SQLContext
from pyspark.sql import SQLContext, functions as F, SparkSession

spark_session = SparkSession.builder.appName('P8').getOrCreate()
# Si besoin d'effectuer des requetes SQL sur un DF:
# sc = spark_session.sparkContext
# sqlContext = SQLContext(sc)

dept = [("Finance",10),
        ("Marketing",20),
        ("Sales",30),
        ("IT",40)
      ]


deptColumns = ["dept_name","dept_id"]
deptDF = spark_session.createDataFrame(data=dept, schema = deptColumns)
deptDF.printSchema()
deptDF.show(truncate=False)


deptDF.show(10)

# https://spark.apache.org/docs/latest/ml-features.html
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark_session.createDataFrame(data, ["features"])

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
