from pyspark import SparkContext
import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql import HiveContext
from pyspark.sql import functions as sf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Row
from functools import partial
from pyspark.ml.regression import LinearRegression
import string
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

#!pip install -U nltk

#print "bonjour Axel"

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


def tokenize(x):
    try:
        p=set(string.punctuation)
        doc=''.join([c for c in str(x.encode("UTF-8")).lower() if c not in p ])
        words=doc.split()
        doc =[ word for word in words if word not in stopwords  ]
        stemmer=PorterStemmer()
        for i,word in enumerate(doc):
            doc[i]=stemmer.stem(word.decode('UTF-8'))
        return ' '.join(doc)
    except:
        return ''

def fixEncoding(x):
    # fix encoding in fields name and value
    id = x['product_uid']
    name = ''
    if x['name'] is not None:
        name = x['name'].encode("UTF-8")
    value = ""
    if x['value'] is not None:
        value = x['value'].encode("UTF-8")
    retVal = '%s %s.' % (name, value)
    # return tuple instead of row
    return (id, [retVal])

def fixEncodingDescription(x):
    # fix encoding in fields name and value
    id = x['product_uid']
    product_description = ''
    if x['product_description'] is not None:
        product_description = x['product_description'].encode("UTF-8")
    # return tuple instead of row
    return (id, [product_description])




def addFeatureLen(row):
    vector = row['tf_idf']
    size = vector.size
    newVector = {}
    for i, v in enumerate(vector.indices):
        newVector[v] = vector.values[i]
    newVector[size] = len(vector.indices)
    size += 1
    # we cannot change the input Row so we need to create a new one
    data = row.asDict()
    data['tf_idf'] = SparseVector(size, newVector)
    # new Row object with specified NEW fields
    newRow = Row(*data.keys())
    # fill in the values for the fields
    newRow = newRow(*data.values())
    return newRow

def addFeatureClean(row):
	vector=row['product_title_clean']
	size=vector.size
	newVector={}
	for i,v in enumerate(vector.indices):
		newVector[v]=vector.values[i]
	newVector[size]=len(vector.indices)
	size+=1
	#we cannot change the input Row so we need to create a new one
	data=row.asDict()
	data['product_title_clean']= SparseVector(size,newVector)
	#new Row object with specified NEW fields
	newRow=Row(*data.keys())
	#fill in the values for the fields
	newRow=newRow(*data.values())
	return newRow

def cleanData(row, model):
    # we are going to fix search term field
    text = row['search_term'].split()
    for i, v in enumerate(text):
        text[i] = correct(v, model)
    data = row.asDict()
    # create new field for cleaned version
    data['search_term2'] = text
    newRow = Row(*data.keys())
    newRow = newRow(*data.values())
    return newRow


def newFeatures(row):
    vector = row['tf_idf']
    data = row.asDict()
    data['features'] = DenseVector([len(vector.indices), vector.values.min()])
    newRow = Row(*data.keys())
    newRow = newRow(*data.values())
    return newRow


sc = SparkContext(appName="Example1")


tokenize_udf = udf(tokenize,StringType())

sqlContext = HiveContext(sc)
counter = 0
print "###############"
# READ data
data = sqlContext.read.format("com.databricks.spark.csv"). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/dssp/datacamp/train.csv").repartition(100)
print "data loaded - head:"
print data.head(5)
print "################"

print "add new column################"
#data = sqlContext.createDataFrame(data.withColumn('product_title_clean', tokenize_udf(data["product_title"])).rdd)
#data = data.withColumn('product_title_clean', tokenize_udf(data["product_title"]))


#data =sqlContext.createDataFrame(data.rdd.map(lambda row:Row(row.__fields__ + ["product_title_clean"])(row + (tokenize_udf(row.product_title), ))))

print "test clean data################"
print data.head(5)




#JOIN ON PRODUCT DEFINITION

descritpion = sqlContext.read.format("com.databricks.spark.csv"). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/dssp/datacamp/product_descriptions.csv").repartition(100)

print "descritpion loaded - head:"
print descritpion.head()
print "################"

# attributes: 0-N lines per product
# Step 1 : fix encoding and get data as an RDD (id,"<attribute name> <value>")
descRDD = descritpion.rdd.map(fixEncodingDescription)
print "new RDD:"
print descRDD.first()
print "################"
# Step 4 join data
fulldata = data.join(descRDD, ['product_uid'], 'left_outer')
print "Joined Data:"
print fulldata.head()
print "################"




#JOIN ON ATTRIBUTES

attributes = sqlContext.read.format("com.databricks.spark.csv"). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/dssp/datacamp/attributes.csv").repartition(100)

print "attributes loaded - head:"
print attributes.head()
print "################"

# attributes: 0-N lines per product
# Step 1 : fix encoding and get data as an RDD (id,"<attribute name> <value>")
attRDD = attributes.rdd.map(fixEncoding)
print "new RDD:"
print attRDD.first()
print "################"
# Step 2 : group attributes by product id
attAG = attRDD.reduceByKey(lambda x, y: x + y).map(lambda x: (x[0], ' '.join(x[1])))
print "Aggregated by product_id:"
print attAG.first()
print "################"
# Step 3 create new dataframe from aggregated attributes
atrDF = sqlContext.createDataFrame(attAG, ["product_uid", "attributes"])
print "New dataframe from aggregated attributes:"
print atrDF.head()
print "################"
# Step 4 join data
fulldata = data.join(atrDF, ['product_uid'], 'left_outer')
print "Joined Data:"
print fulldata.head()
print "################"

# TF-IDF features
#Step 0 : make one mega text column text_clean
print "Clean title"
fulldata = sqlContext.createDataFrame(fulldata.withColumn('title_clean', tokenize_udf(fulldata["product_title"])).rdd)
print "Clean attribute"
fulldata = sqlContext.createDataFrame(fulldata.withColumn('attribute_clean', tokenize_udf(fulldata["attributes"])).rdd)
print "Clean description"
fulldata = sqlContext.createDataFrame(fulldata.withColumn('description_clean', tokenize_udf(fulldata["product_description"])).rdd)
print "merge cleaning"
fulldata = sqlContext.createDataFrame(fulldata.withColumn('text_clean_temp', sf.concat(sf.col('title_clean'),sf.lit(' '), sf.col('attribute_clean'))).rdd)
fulldata = sqlContext.createDataFrame(fulldata.withColumn('text_clean', sf.concat(sf.col('text_clean_temp'),sf.lit(' '), sf.col('description_clean'))).rdd)
print fulldata.head()

# Step 1: split text field into words
tokenizer = Tokenizer(inputCol="text_clean", outputCol="words_title")
fulldata = tokenizer.transform(fulldata)
print "Tokenized Title:"
print fulldata.head()
print "################"
# Step 2: compute term frequencies
hashingTF = HashingTF(inputCol="words_title", outputCol="tf")
fulldata = hashingTF.transform(fulldata)
print "TERM frequencies:"
print fulldata.head()
print "################"
# Step 3: compute inverse document frequencies
idf = IDF(inputCol="tf", outputCol="tf_idf")
idfModel = idf.fit(fulldata)
fulldata = idfModel.transform(fulldata)
print "IDF :"
print fulldata.head()
print "################"

# Step 4 new features column / rename old
fulldata = sqlContext.createDataFrame(fulldata.rdd.map(newFeatures))
print "NEW features column :"
print fulldata.head()
print "################"
# Step 5: ALTERNATIVE ->ADD column with number of terms as another feature
fulldata = sqlContext.createDataFrame(fulldata.rdd.map(addFeatureLen))  # add an extra column to tf features
fulldata = fulldata.withColumnRenamed('tf_idf', 'tf_idf_plus')
print "ADDED a column and renamed :"
print fulldata.head()
print "################"

# create NEW features & train and evaluate regression model
# Step 1: create features
fulldata = fulldata.withColumnRenamed('relevance', 'label').select(['label', 'features'])

# Simple evaluation : train and test split
(train, test) = fulldata.rdd.randomSplit([0.8, 0.2])

# Initialize regresion model
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(sqlContext.createDataFrame(train))

# Apply model to test data
result = lrModel.transform(sqlContext.createDataFrame(test))
# Compute mean squared error metric
MSE = result.rdd.map(lambda r: (r['label'] - r['prediction']) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

