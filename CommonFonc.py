
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
	text=row['search_term'].split()
	for i,v in enumerate(text):
		text[i]=correct(v,model)
	data=row.asDict()
	#create new field for cleaned version
	data['search_term2']= text
	newRow=Row(*data.keys())
	newRow=newRow(*data.values())
	return newRow


