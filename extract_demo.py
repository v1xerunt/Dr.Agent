<<<<<<< HEAD
import os
from os import listdir
from os.path import isfile, join
import shutil

TASK = 'decompensation'

mypath = './data/'+TASK+'/train/'
for f in listdir(mypath):
    for ep in range(1, 100):
        file_name = mypath+str(f)+'/episode'+str(ep)+'.csv'
        targe_name = './data/'+TASK+'/demographic/' + str(f) + '_episode' + str(ep) + '.csv'
        if os.path.exists(file_name):
            shutil.copyfile(file_name, targe_name)

mypath = './data/'+TASK+'/test/'
for f in listdir(mypath):
    for ep in range(1, 100):
        file_name = mypath+str(f)+'/episode'+str(ep)+'.csv'
        targe_name = './data/'+TASK+'/demographic/' + str(f) + '_episode' + str(ep) + '.csv'
        if os.path.exists(file_name):
            shutil.copyfile(file_name, targe_name)
=======
import os
from os import listdir
from os.path import isfile, join
import shutil

TASK = 'decompensation'

mypath = './data/'+TASK+'/train/'
for f in listdir(mypath):
    for ep in range(1, 100):
        file_name = mypath+str(f)+'/episode'+str(ep)+'.csv'
        targe_name = './data/'+TASK+'/demographic/' + str(f) + '_episode' + str(ep) + '.csv'
        if os.path.exists(file_name):
            shutil.copyfile(file_name, targe_name)

mypath = './data/'+TASK+'/test/'
for f in listdir(mypath):
    for ep in range(1, 100):
        file_name = mypath+str(f)+'/episode'+str(ep)+'.csv'
        targe_name = './data/'+TASK+'/demographic/' + str(f) + '_episode' + str(ep) + '.csv'
        if os.path.exists(file_name):
            shutil.copyfile(file_name, targe_name)
>>>>>>> 742f7a833bbc02da15cd7752dc1d2cdd8606aeb1
