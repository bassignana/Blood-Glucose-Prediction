#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:26:32 2020

@author: tommasobassignana
"""
import numpy as np
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

# y = np.array([1, 2, 3, 4, 5, 6])
# tscv = expanding_window()
# for train_index, test_index in tscv.split(X):
#     print(train_index)
#     print(test_index)



initial= 1
horizon = 1
period = 1
gap = 1
 
data = X
counter = 0 # for us to iterate and track later, like i?
data

data_length = data.shape[0] # rows 
print(data_length)
data_index = list(np.arange(data_length))
print(data_index)
   
output_train = []
output_test = []

# append initial index
output_train.append(list(np.arange(initial)))
print(output_train)

progress = [x for x in data_index if x not in list(np.arange(initial))] #indexes left to append to train 
print(progress)


output_train[counter]
output_test.append([x for x in data_index if x not in output_train[counter]][:horizon])
print(output_test)

        # clip initial indexes from progress since that is what we are left 

        
while len(progress) != 0:
            print(" len progress is")
            print(len(progress))
            print("progress is")
            print(progress)
            temp = progress[:period]
            print("period is")
            print(period)
            print("temp is")
            print(temp)
            
            print("counter is")
            print(counter)
            print("output_train[counter]")
            print(output_train[counter])
            to_add = output_train[counter] + temp
            print("to add is")
            print(to_add)
            # update the train index 
            print("train_index before adding")
            print(output_train)
            output_train.append(to_add)
            print("train_index after adding")
            print(output_train)
            # increment counter 
            counter +=1 
            # then we update the test index 
            print("test_index before adding")
            print(output_test)
            print("what to add to test")
            #to_add_test = [x for x in data_index if x not in output_train[counter] ][:(horizon + gap)]
            to_add_test = [x for x in data_index if x not in output_train[counter] ][:horizon]
            if len(to_add_test) == 0:
                break
            to_add_test = int(to_add_test[-1])+gap
            print(to_add_test)
            output_test.append(to_add_test)
            print("test_index after adding")
            print(output_test)

            # update progress 
            progress = [x for x in data_index if x not in output_train[counter]]	
            
# clip the last element of output_train and output_test
#output_train = output_train[:-1]
#output_test = output_test[:-1]
output_train = output_train[:-gap]
output_test = output_test[:-gap]
print("final")
print(output_train)
print(output_test)
# mimic sklearn output 
#index_output = [(train,test) for train,test in zip(output_train,output_test)]
        
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

# y = np.array([1, 2, 3, 4, 5, 6])
# tscv = expanding_window()
# for train_index, test_index in tscv.split(X):
#     print(train_index)
#     print(test_index)



initial= 1
horizon = 1
period = 1
gap = 0

#è problematico per gap = 0 e gap = 1??

data = X
counter = 0 # for us to iterate and track later, like i?
data

data_length = data.shape[0] # rows 
print(data_length)
data_index = list(np.arange(data_length))
print(data_index)
   
output_train = []
output_test = []

# append initial index
output_train.append(list(np.arange(initial)))
print(output_train)

progress = [x for x in data_index if x not in list(np.arange(initial))] #indexes left to append to train 
print(progress)


output_train[counter]
output_test.append([x for x in data_index if x not in output_train[counter]][:horizon])
print(output_test)

        # clip initial indexes from progress since that is what we are left 

        
while len(progress) != 0:
            print(" len progress is")
            print(len(progress))
            print("progress is")
            print(progress)
            temp = progress[:period]
            print("period is")
            print(period)
            print("temp is")
            print(temp)
            
            print("counter is")
            print(counter)
            print("output_train[counter]")
            print(output_train[counter])
            to_add = output_train[counter] + temp
            print("to add is")
            print(to_add)
            # update the train index 
            print("train_index before adding")
            print(output_train)
            output_train.append(to_add)
            print("train_index after adding")
            print(output_train)
            # increment counter 
            counter +=1 
            # then we update the test index 
            print("test_index before adding")
            print(output_test)
            print("what to add to test")
            #to_add_test = [x for x in data_index if x not in output_train[counter] ][:(horizon + gap)]
            to_add_test = [x for x in data_index if x not in output_train[counter] ][:horizon]
            if len(to_add_test) == 0:
                break
            to_add_test = int(to_add_test[-1])+gap
            print(to_add_test)
            output_test.append(to_add_test)
            print("test_index after adding")
            print(output_test)

            # update progress 
            progress = [x for x in data_index if x not in output_train[counter]]	
            
# clip the last element of output_train and output_test
#output_train = output_train[:-1]
#output_test = output_test[:-1]
if gap != 0:
    output_train = output_train[:-gap]
    output_test = output_test[:-gap]
    output_train = output_train[:len(output_test)]
print("final")
print(output_train)
print(output_test)

############
       
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4],[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])



# initial= 1
# horizon = 1
# period = 1
# gap = 0

# #è problematico per gap = 0 e gap = 1??

# data = X
# counter = 0 # for us to iterate and track later, like i?
# data

# data_length = data.shape[0] # rows 
# print(data_length)
# data_index = list(np.arange(data_length))
# print(data_index)
   
# output_train = []
# output_test = []
# real_output_train = []
# # append initial index
# output_train.append(list(np.arange(initial)))
# print(output_train)

# progress = [x for x in data_index if x not in list(np.arange(initial))] #indexes left to append to train 
# print(progress)


# output_train[counter]
# output_test.append([x for x in data_index if x not in output_train[counter]][:horizon])
# print(output_test)

#         # clip initial indexes from progress since that is what we are left 

# cut = period
        
# while len(progress) != 0:
#             print(" len progress is")
#             print(len(progress))
#             print("progress is")
#             print(progress)
#             temp = progress[:period]#period è quanti indici aggiungo ogni volta
#             print("period is")
#             print(period)
#             print("temp is")
#             print(temp)
#             print("counter is")
#             print(counter)
#             print("output_train[counter]")
#             print(output_train[counter])
#             to_add = output_train[counter] + temp
#             print("QUELLO CHE VORREI")
#             print("cut is")
#             print(cut)
#             print(to_add[cut:])
#             cut += 1
#             print("to add is")
#             print(to_add)
#             # update the train index 
#             print("train_index before adding")
#             print(output_train)
#             output_train.append(to_add)
#             print("train_index after adding")
#             print(output_train)
#             # increment counter 
#             counter +=1 
#             # then we update the test index 
#             print("test_index before adding")
#             print(output_test)
#             print("what to add to test")
#             #to_add_test = [x for x in data_index if x not in output_train[counter] ][:(horizon + gap)]
#             to_add_test = [x for x in data_index if x not in output_train[counter] ][:horizon]
#             if len(to_add_test) == 0:
#                 break
#             to_add_test = int(to_add_test[-1])+gap
#             print(to_add_test)
#             output_test.append(to_add_test)
#             print("test_index after adding")
#             print(output_test)

#             # update progress 
#             progress = [x for x in data_index if x not in output_train[counter]]	
            
# # clip the last element of output_train and output_test
# #output_train = output_train[:-1]
# #output_test = output_test[:-1]
# if gap != 0:
#     output_train = output_train[:-gap]
#     output_test = output_test[:-gap]
#     output_train = output_train[:len(output_test)]
# print("final")
# print(output_train)
# print(output_test)