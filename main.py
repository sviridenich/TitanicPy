import os
import json
import csv as csv
import numpy as np

def get_paths():
    paths = json.loads(open("Settings.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def upload_data(path):
    '''Given the path to the csv file, return the array of the data
    (str) -> array '''

    csv_file_object = csv.reader(open(path, 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    data = [] #Creat a variable called
    for row in csv_file_object: #Skip through each row in the csv file
        data.append(row) #adding each row to the data variable
    data = np.array(data) #Then convert from a list to an array
    return data

def converting_str_to_int(data_array, n_sex, n_embark):
    ''' Given the data array and index of collumns with sex and embarking, return modifed array'''    
    #Male = 1, female = 0: 
    data_array[data_array[0::,n_sex]=='male',n_sex] = 1
    data_array[data_array[0::,n_sex]=='female',n_sex] = 0

    #embark c=0, s=1, q=2: 
    data_array[data_array[0::,n_embark] =='C',n_embark] = 0
    data_array[data_array[0::,n_embark] =='S',n_embark] = 1
    data_array[data_array[0::,n_embark] =='Q',n_embark] = 2

    return data_array

def imputing_median(data_array, n):
    '''Given the array and column index, we substitute median value for missing ones'''
    data_array[data_array[0::,n] == '',n] = np.median(data_array[data_array[0::,n]!= '',n].astype(np.float))
    return data_array

def imputing_most_common(data_array, n):
    '''Given the array and columnd index we substitute most common values for missing ones, although the algorithm got bug'''
    data_array[data_array[0::,n] == '',n] = np.round(np.mean(data_array[data_array[0::,n]!= '',n].astype(np.float)))
    return data_array

def imputing_median_by_the_group(data_array, n_to_mutate, n_based_on ):
    '''Given the array and index collumns,we substitue median values in first column, based on grouping by second column provided'''
    for i in xrange(np.size(data_array[0::,n_based_on])):
        if data_array[i,n_to_mutate] == '':
            data_array[i,n_to_mutate] = np.median(data_array[(test_data[0::,n_to_mutate] != '') & \
                                                 (data_array[0::,n_based_on] == data_array[i,n_based_on]),n_to_mutate].astype(np.float))
    return data_array

TRAIN_DATA_PATH = get_paths()["train_data_path"]
TEST_DATA_PATH = get_paths()["test_data_path"]
    
#Load in the training
train_data = upload_data(TRAIN_DATA_PATH)


### column names in train data
col_sex = 3
col_embark = 10
col_age = 4

col_price = 8
col_class = 1

col_name = 2
col_cabin = 7
col_ticket = 9


#I need to convert all strings to integer classifiers:
train_data = converting_str_to_int( train_data ,n_sex = col_sex , n_embark = col_embark )

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
train_data = imputing_median(train_data, col_age)

#All missing ebmbarks just make them embark from most common place
train_data = imputing_most_common(train_data, col_embark )

#All the missing prices assume median of their respectice class
train_data = imputing_median_by_the_group (train_data , col_price, col_class)

#remove the name data, cabin and ticket
train_data = np.delete(train_data,[col_name,col_cabin,col_ticket],1) 


### columns in test data is shift by 1
col_sex -= 1
col_embark -= 1
col_age -= 1

col_price -= 1
col_class -= 1

col_name -= 1
col_cabin -= 1
col_ticket -= 1

#I need to do the same with the test data now so that the columns are in the same
#as the training data

test_data = upload_data(TEST_DATA_PATH)
#I need to convert all strings to integer classifiers:
test_data = converting_str_to_int( test_data ,n_sex = col_sex , n_embark = col_embark )
# age
test_data = imputing_median(test_data, col_age)
# embarks 
test_data = imputing_most_common(test_data, col_embark )
# prices 
test_data = imputing_median_by_the_group (test_data , col_price, col_class)
test_data = np.delete(test_data,[col_name,col_cabin,col_ticket],1) 

#The data is now ready to go. So lets train then test!
