import string
import nltk
from nltk.stem.porter import *

#!pip install -U nltk


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
def rmP(x):
    p=set(string.punctuation)
    doc=''.join([c for c in str(x).lower() if c not in p ])
    words=doc.split()
    doc =[ word for word in words if word not in stopwords  ]
    stemmer=PorterStemmer()
    for i,word in enumerate(doc):
        doc[i]=stemmer.stem(word.decode('utf-8'))
    return ' '.join(doc)





def fixEncoding(x):
    # fix encoding in fields name and value id=x[ 'product_uid']
    name=''
    if x['name'] is not None:
        name=x[ 'name'].encode("UTF-8")
    value=""
    if x['value'] is not None:
        value=x[ 'value'].encode("UTF-8")
    retVal= '%s %s.'%(name, value)
    # return tuple instead of row
    return (id,[retVal] )


def cleanData(row,model):
	#we are going to fix search term field
	text=rmP(row['search_term']).split()
	for i,v in enumerate(text):
		text[i]=correct(v,model)
	data=row.asDict()
	#create new field for cleaned version
	data['search_term2']= text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow


