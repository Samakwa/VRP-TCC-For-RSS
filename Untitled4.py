#!/usr/bin/env python
# coding: utf-8

# In[30]:


import csv
transposed = [] 

i=0
transposed_row=[]

with open('Route_Distances.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    
    next(csvReader)
    for row in csvReader:
       if row[0]!=str(i):
         c=transposed_row.copy()
         transposed.append(c)
         i=i+1
         del transposed_row[:]
         transposed_row.append(float(row[5]))
         
       else:
            
            transposed_row.append(float(row[5]))
            
      
      


     
       #del transposed_row[:]
       #transposed_row.append(row[5])
       #i=i+1
      
c=transposed_row.copy()
transposed.append(c)
del transposed_row[:]       
      
for item in transposed:

    print(item)


# In[ ]:





# In[ ]:




