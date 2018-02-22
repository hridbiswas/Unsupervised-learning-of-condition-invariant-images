# Unsupervised-learning-of-condition-invariant-images
#My work is inspired by the paper: "Unsupervised Online Learning of Condition-Invariant Images for Place Recognition" 

# In this task, initially Alderley day/night dataset is used 
[https://wiki.qut.edu.au/pages/viewpage.action?pageId=181178395]
#Training data: 2500 images are used; 1250 images are taken from 'Day' folder, rest 1250 images are from 'Night' folder.
#GrayScale images are subsampled to 26*64 pixels
#Size of traing dataset 2500*1664
#PCA is applied on training dataset
###### obtained scores and eigenvectors/eigenfaces

#After getting scores are eigenvectors, those are used to transform our test images
#To make our transformation Invariant, first 100 principle components are discarded and laters (e.g. 101:200) principle components are taken

# for test, one one pair of images are shown here (which can be applied to the whole test folder).
Test images:
Day_12180.jpg
Night_14366.jpg


## Independent Component Analysis method also applied on this dataset. But it doesn't perform well here.


