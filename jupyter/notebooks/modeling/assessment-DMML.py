#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import stats
import math


# # 1. CSV to ARFF

# In[ ]:


# Since we are using Python, we do  not need to complete this step


# # Loading Data

# ## pandas.read_csv

# In[ ]:

file_path = "../../data/raw/"

X = pd.read_csv(f"{file_path}x_train_gr_smpl.csv", delimiter=',')
X.head()

# In[ ]:
Y = pd.read_csv(f"{file_path}y_train_smpl.csv", delimiter=',')
Y.columns = ['target']
y0 = pd.read_csv(f"{file_path}y_train_smpl_0.csv", delimiter=',')
y0.columns = ['target']
y1 = pd.read_csv(f"{file_path}y_train_smpl_1.csv", delimiter=',')
y1.columns = ['target']
y2 = pd.read_csv(f"{file_path}y_train_smpl_2.csv", delimiter=',')
y2.columns = ['target']
y3 = pd.read_csv(f"{file_path}y_train_smpl_3.csv", delimiter=',')
y3.columns = ['target']
y4 = pd.read_csv(f"{file_path}y_train_smpl_4.csv", delimiter=',')
y4.columns = ['target']
y5 = pd.read_csv(f"{file_path}y_train_smpl_5.csv", delimiter=',')
y5.columns = ['target']
y6 = pd.read_csv(f"{file_path}y_train_smpl_6.csv", delimiter=',')
y6.columns = ['target']
y7 = pd.read_csv(f"{file_path}y_train_smpl_7.csv", delimiter=',')
y7.columns = ['target']
y8 = pd.read_csv(f"{file_path}y_train_smpl_8.csv", delimiter=',')
y8.columns = ['target']
y9 = pd.read_csv(f"{file_path}y_train_smpl_9.csv", delimiter=',')
y9.columns = ['target']
Y.tail()

# # Create datasets

# In[ ]:


train_smpl_0 = pd.concat([X, y0], axis=1)
train_smpl_1 = pd.concat([X, y1], axis=1)

# In[ ]:


train_smpl_2 = pd.concat([X, y2], axis=1)
train_smpl_3 = pd.concat([X, y3], axis=1)

# In[ ]:


train_smpl_4 = pd.concat([X, y4], axis=1)
train_smpl_5 = pd.concat([X, y5], axis=1)

# In[ ]:


train_smpl_6 = pd.concat([X, y6], axis=1)
train_smpl_7 = pd.concat([X, y7], axis=1)

# In[ ]:


train_smpl_8 = pd.concat([X, y8], axis=1)
train_smpl_9 = pd.concat([X, y9], axis=1)

# In[ ]:


train_smpl = pd.concat([X, Y], axis=1)
# train_smpl.head()

# In[ ]:


# train_smpl_0.head()

# # class distribution

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# train_smpl.hist(column='target')


# In[ ]:


train_smpl.target.value_counts()

# #### imbalanced classes => won't generalize

# # 2. Data Randomisation

# ## sklearn.utils.shuffle

# In[ ]:
train_smpl = shuffle(train_smpl, random_state=42)
train_smpl_0 = shuffle(train_smpl_0, random_state=42)
train_smpl_1 = shuffle(train_smpl_1, random_state=42)
train_smpl_2 = shuffle(train_smpl_2, random_state=42)
train_smpl_3 = shuffle(train_smpl_3, random_state=42)
train_smpl_4 = shuffle(train_smpl_4, random_state=42)
train_smpl_5 = shuffle(train_smpl_5, random_state=42)
train_smpl_6 = shuffle(train_smpl_6, random_state=42)
train_smpl_7 = shuffle(train_smpl_7, random_state=42)
train_smpl_8 = shuffle(train_smpl_8, random_state=42)
train_smpl_9 = shuffle(train_smpl_9, random_state=42)
train_smpl.head()


# In[ ]:


# train_smpl.info()

# In[ ]:


# train_smpl.describe()

# # 3. Reducing the size

# #### Because the data runs the data as a Python file, we do not need to reduce the size of our data set. 
# 
# 

# # 4 - 6: NB, Features/Attributes Selection, & Improving #5's Classification  

# ## correlation matrix

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')

# plt.matshow(train_smpl.corr())
# plt.show()


# In[ ]:


# # Test Train Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_smpl[train_smpl.columns[:2303]],
                                                    train_smpl['target'],
                                                    test_size=0.33, random_state=42)

X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(train_smpl_20[train_smpl_20.columns[:20]], train_smpl_20['target'], test_size=0.33, random_state=42)
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(train_smpl_50[train_smpl_50.columns[:50]], train_smpl_50['target'], test_size=0.33, random_state=42)
X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(train_smpl_100[train_smpl_100.columns[:100]], train_smpl_100['target'], test_size=0.33, random_state=42)

# # Modeling

# ## Multinomial Naive Bayes model (multi-class classifier)

# ### Before Features/Attributes Selection

# In[ ]:


clf = MultinomialNB()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)


# In[ ]:

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    https://stackoverflow.com/a/50386871
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:


# plot_confusion_matrix(conf_mat, target_names=y_test.unique())


# In[ ]:


# looking for devious class labels that take high numbers of misclassifications
# for i in range(conf_mat.shape[1]):
# 	column = conf_mat.T[i]
# 	misclassifications = column.sum() - column[i]
# 	print(misclassifications)


# ###4. Because we are not using Weka, we did not need to apply any
# filters to the data before running Naive Bayes


# In[ ]:
print("\n#4 Multilabel - All Features: \n __________________________________")

clf = MultinomialNB()
clf.fit(X_train, y_train)
print(f"Multinomial NB: {clf.score(X_test, y_test)} ")

# ### 5. For each of the 10 train_smpl_label files, record the first 10 fields, in order of the absolute correlation
# value for each street sign.

# In[ ]:

print("\n#5\n __________________________________")

fileList = [train_smpl_0, train_smpl_1, train_smpl_2, train_smpl_3, train_smpl_4,
            train_smpl_5, train_smpl_6, train_smpl_7, train_smpl_8, train_smpl_9]
for f, file in enumerate(fileList):
    corrArr = []
    for j in file.columns[:-1]:
        corrVal, pVal = stats.pearsonr(file.iloc[:, int(j)], file.iloc[:, -1])
        # Record the absolute value of the correlation
        corrArr.append(math.fabs(corrVal))
    print(f"File {f}:")
    print(sorted([(x, i) for (i, x) in enumerate(corrArr)], reverse=True)[:10])

print("\n#6\n __________________________________")

# ### 6. After Features/Attributes Selection (20/50/100)

def run_bernoulli_nb(data, num_feat, class_num):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data[data.columns[:2303]], data['target'],
                                                                test_size=0.33, random_state=42)

    clf = BernoulliNB()
    clf.fit(X_train_2, y_train_2)
    print(f"Bernoulli NB Best {num_feat}0 Class {class_num}: {clf.score(X_test_2, y_test_2)}")

def run_multi_nb_binary(data, num_feat, class_num):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data[data.columns[:2303]], data['target'],
                                                                test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train_2, y_train_2)
    print(f"Multinomial for Binary NB Best {num_feat}0: Class {class_num}: {clf.score(X_test_2, y_test_2)}")

def run_multi_nb(data, num_feat):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data[data.columns[:2303]], data['target'],
                                                                test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train_2, y_train_2)
    print(f"Multinomial NB Best {num_feat}0: {clf.score(X_test_2, y_test_2)}")


# In[ ]:

# Select top 2, 5, and 10 features for all train_smpl_<label> files
# to create best 20, 50, and 100 sets for all 10 classes

bestSet = [2, 5, 10]

for i, num in enumerate(bestSet):
    k_best_list = pd.DataFrame()
    # Get top k for each 10 classes
    for f, file in enumerate(fileList):
        k_best = SelectKBest(chi2, k=num).fit_transform(file[file.columns[:2303]], file['target'])
        k_best = pd.DataFrame(k_best)
        k_best_list = pd.concat([k_best_list, k_best], axis=1)
        # print(k_best_list.head())

    print(f"\nBinary NB Classifier {num}0:\n __________________________________")
    # run binary NB classifiers for each class
    for f, file in enumerate(fileList):
        k_best_binary = pd.concat([k_best_list, file['target'].reset_index()], axis=1)
        # print(f"K BEST BINARY: {k_best_binary}")
        run_multi_nb_binary(k_best_binary, num, f)

    print(f"\nMultilabel NB Classifier {num}0:\n __________________________________")
    k_best_multi = pd.concat([k_best_list, train_smpl['target'].reset_index()], axis=1)
    # print(f"K BEST MULTI: {k_best_multi}")
    run_multi_nb(k_best_multi, num)

# ## Gaussian Naive Bayes model (mono-class classifier)


# In[ ]:


# Multinomial Nb is more accurate than Gaussian Nb ( 0.75 > 0.69 )


# # Analysis for 4 & 7

# ### 4. Explain the reason for choosing and using these filters. Once you can run the algorithm, record, compare and
# analyse the classifier’s accuracy on different classes (as given by the Weka Summary and the confusion matrix).

# #### Since we are not using Weka, we did not need to apply any filters to the data before running Naive Bayes.

# ### 7. What kind of information about this data set did you learn, as a result of the above experiments? You should
# ask questions such as: Which streets signs are harder to recognise? Which street signs are most easily confused?
# Which attributes (fields) are more reliable and which are less reliable in classification of street signs? What was
# the purpose of Tasks 5 and 6? What would happen if the data sets you used in Tasks 4, 5 and 6 were not randomised?
# What would happen if there is cross-correlation between the non-class attributes? You will get more marks for more
# interesting and ``out of the box” questions and answers. Explain your conclusions logically and formally,
# using the material from the lecture notes and from your own reading to interpret the results that Weka produces.

# #### Analysis....

# ## Conclusions from initial experiments (q7)

# Using the multilabel confusion matrix, it is apparent that the multinomial Naïve Bayes classifier struggled with
# some classes, predicting them incorrectly most of the time.
# 
# In particular, class 8 was more often classified as class 1 (probability 0.278), class 7 (probability 0.2023) and
# class 6 (probability 0.1974) before itself (probability 0.1283).
# 
# //It looks like the classifier was confused by ...
# 
# 
# 
# The strongest misclassification in the matrix is true members of class 3 being classified as class 7 (probability
# 0.2023). However, there is relatively very little of the reverse misclassification (true class 7 predicted as class
# 3).
#
# //^^ investigate this?
# 
# 
# 
# Class 7 punches above its weight as the most abundant misclassification which is perhaps due to the relative
# infrequency of data for class 7. There must be some other factor, as there are less populated classes.
# 
#
