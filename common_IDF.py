from pyspark import SparkContext
import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext 
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer,CountVectorizer
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row
from functools import partial
from pyspark.ml.regression import LinearRegression

def fixEncoding(x):
	# fix encoding in fields name and value
	id=x['product_uid']
	name=''
	if x['name'] is not None:
		name=x['name'].encode("UTF-8")
	value=""
	if x['value'] is not None:
		value=x['value'].encode("UTF-8")
	retVal='%s %s.'%(name,value)
	#return tuple instead of row 
	return (id,[retVal])
	
def addFeatureLen(row):
	vector=row['tf_idf']
	size=vector.size
	newVector={}
	for i,v in enumerate(vector.indices):
		newVector[v]=vector.values[i]
	newVector[size]=len(vector.indices)
	size+=1
	#we cannot change the input Row so we need to create a new one
	data=row.asDict()
	data['tf_idf']= SparseVector(size,newVector)
	#new Row object with specified NEW fields
	newRow=Row(*data.keys())
	#fill in the values for the fields
	newRow=newRow(*data.values())
	return newRow
	
def cleanData(row,model):
	#we are going to fix search term field
	text=row['search_term'].split()
	for i,v in enumerate(text):
		text[i]=correct(v,model)
	data=row.asDict()
	#create new field for cleaned version
	data['search_term2']= text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow
	
	
def newFeatures(row):
	vector=row['tf_idf']
	data=row.asDict()
	data['features']= DenseVector([len(vector.indices),vector.values.min()])
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow


sc = SparkContext(appName="Example1")

sqlContext = HiveContext(sc)
print "###############"
#READ data
data=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load("/dssp/datacamp/train.csv").repartition(100)
print "data loaded - head:"	
print data.head()
print "################"

attributes=sqlContext.read.format("com.databricks.spark.csv").\
	option("header", "true").\
	option("inferSchema", "true").\
	load("/dssp/datacamp/attributes.csv").repartition(100)
	
print "attributes loaded - head:"	
print attributes.head()
print "################"

#attributes: 0-N lines per product
#Step 1 : fix encoding and get data as an RDD (id,"<attribute name> <value>")
attRDD=attributes.rdd.map(fixEncoding)
print "new RDD:"
print attRDD.first()
print "################"
#Step 2 : group attributes by product id
attAG=attRDD.reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],' '.join(x[1])))
print "Aggregated by product_id:"
print attAG.first()
print "################"
#Step 3 create new dataframe from aggregated attributes
atrDF=sqlContext.createDataFrame(attAG,["product_uid", "attributes"])
print "New dataframe from aggregated attributes:"
print atrDF.head()
print "################"
#Step 4 join data
fulldata=data.join(atrDF,['product_uid'],'left_outer')
print "Joined Data:"
print fulldata.head()
print "################"

fulldata.registerTempTable("df")
concatedField=sqlContext.sql("SELECT product_uid,attributes,product_title,search_term,CONCAT(product_title,' ',search_term) as allText FROM df")
print concatedField.head()


tokenizer = Tokenizer(inputCol="allText", outputCol="words_allText")
concatedField = tokenizer.transform(concatedField)
tokenizer = Tokenizer(inputCol="search_term", outputCol="words_search_term")
concatedField = tokenizer.transform(concatedField)

hashingTF = HashingTF(inputCol="words_allText", outputCol="tf1")
concatedField = hashingTF.transform(concatedField)
hashingTF = HashingTF(inputCol="words_search_term", outputCol="tf2")
concatedField = hashingTF.transform(concatedField)
print concatedField.head()

print "TERM frequencies:"
print concatedField.head()
print "################"
#Step 3: compute inverse document frequencies
idf = IDF(inputCol="tf1", outputCol="tf_idf1")
idfModel = idf.fit(concatedField)
concatedField = idfModel.transform(concatedField)
concatedField=concatedField.withColumnRenamed('tf_idf1', 'tf_idf_all')
concatedField=concatedField.withColumnRenamed('tf1', 'tf1_all')
concatedField=concatedField.withColumnRenamed('tf2', 'tf1')
concatedField = idfModel.transform(concatedField)

print concatedField.head()




