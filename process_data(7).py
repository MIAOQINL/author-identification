#-----------------------------------------------------#
# Program: process_data.py
# Status: working on it
# Programmer: Joe
# Purpose: Read and process data from train/test.
#
# Input: please see below
#
# Output: NA
#
#-----------------------------------------------------#

# Packages
import numpy as np
from Ngram import NGram
import csv
#-----------------------#
# Read Data
#-----------------------#


# Input Files
train_file = '../RawData/unzip/train_tweets.txt'
test_file = '../RawData/unzip/test_tweets_unlabeled.txt'

# Train
# Open file
with open(train_file,encoding='UTF-8') as train:
	train_data = np.genfromtxt(train,delimiter='\t',dtype='str',comments=None)
	# train_data =np.loadtxt(train,delimiter='\t',dtype='str',usecols=1)
# train_data=train_data[0:100,:]
def nptolist(train_data):
	train_data_dict={}

	
	for data in train_data:
		# data_str=train_data_dict.setdefault(data[0],[])
		train_data_dict.setdefault(data[0],[]).append(''.join(data[1]))
		# train_data_dict[data[0]]=data_str
	
	train_num_list=list(train_data_dict.keys())
	train_data_list=train_data_dict.values()
	return 	train_num_list,	train_data_list


train_num_list,	train_data_list=nptolist(train_data)	
print(len(train_num_list))	
print(len(train_data_list))	
# # Notice
# print(train_file,'is read as np array with shape:',train_data.shape)

# Test
with open(test_file,encoding='UTF-8') as test:
	test_data = test.readlines()
# test_data=test_data[0:100]
print(len(test_data))
# Convert to np array
# test_data = np.array(test_data)

# Notice
# print(test_file,'is read as np array with shape:',test_data.shape)
n_gram5=NGram(items=train_data_list,N=5,pad_char='$')
n_gram4=NGram(items=train_data_list,N=4,pad_char='$')
n_gram3=NGram(items=train_data_list,N=3,pad_char='$')
n_gram2=NGram(items=train_data_list,N=2,pad_char='$')
n_gram1=NGram(items=train_data_list,N=1,pad_char='$')
#-----------------------#
# Process Data
#-----------------------#
# train_ngram  = n_gram.list_ngram()
# print (len(train_ngram))

def predict(test_data,train_num_list):
	predictions=[]
	for query in test_data:
		results=n_gram5.search(query)
		if len(results)>0:
			prediction=results[0][0]
		else:
			results=n_gram4.search(query)
			if len(results)>0:
				prediction=results[0][0]
			else:
				results=n_gram3.search(query)
				if len(results)>0:
					prediction=results[0][0]
				else:
					results=n_gram2.search(query)
					if len(results)>0:
						prediction=results[0][0]
					else:
						results=n_gram1.search(query)
						if len(results)>0:
							prediction=results[0][0]
						else:
							print (query)#[0:min(10,len(query))]
							prediction=0
		predictions.append(train_num_list[prediction])
	return predictions
	
predictions=predict(test_data,train_num_list)
print(len(predictions))	
# Create CSV
header = ['Id','Predicted']

with open('submission.csv', mode='w', newline='') as submission:
    sub_writer = csv.writer(submission, delimiter=',')
    sub_writer.writerow(header)

    for tid in range(len(predictions)):
        sub_writer.writerow([tid+1,predictions[tid]])
