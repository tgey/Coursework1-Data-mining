#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.utils import shuffle


# # 1. CSV to ARFF

# In[ ]:


#Since we are using Python, we do  not need to complete this step


# # Loading Data

# ## pandas.read_csv

# In[12]:


file_path = "../../data/raw/"

X = pd.read_csv(f"{file_path}x_train_gr_smpl.csv", delimiter=',')
X.head()


# In[13]:


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

# In[15]:


train_smpl_0 = pd.concat([X, y0], axis=1)
train_smpl_1 = pd.concat([X, y1], axis=1)


# In[16]:


train_smpl_2 = pd.concat([X, y2], axis=1)
train_smpl_3 = pd.concat([X, y3], axis=1)


# In[17]:


train_smpl_4 = pd.concat([X, y4], axis=1)
train_smpl_5 = pd.concat([X, y5], axis=1)


# In[18]:


train_smpl_6 = pd.concat([X, y6], axis=1)
train_smpl_7 = pd.concat([X, y7], axis=1)


# In[19]:


train_smpl_8 = pd.concat([X, y8], axis=1)
train_smpl_9 = pd.concat([X, y9], axis=1)


# In[20]:


train_smpl = pd.concat([X, Y], axis=1)
train_smpl.head()


# In[21]:


train_smpl_0.head()


# # class distribution

# In[22]:


#get_ipython().run_line_magic('matplotlib', 'inline')
train_smpl.hist(column='target')


# In[23]:


train_smpl.target.value_counts()


# #### imbalanced classes => won't generalize

# # 2. Data Randomisation

# ## sklear.utils.suffle

# In[24]:


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


# In[25]:


# train_smpl.info()


# In[26]:


# train_smpl.describe()


# # 3. Reducing the size

# #### Because the data runs the data as a Python file, we do not need to reduce the size of our data set. 
# 
# 

# # 4 - 6: NB, Features/Attributes Selection, & Improving #5's Classification  

# ## correlation matrix

# In[27]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#plt.matshow(train_smpl.corr())
#plt.show()


# In[28]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
print(f"train_smpl shape:{train_smpl.shape}")

best_20 = SelectKBest(chi2, k=20).fit_transform(train_smpl[train_smpl.columns[:2303]], train_smpl['target'])
best_20 = pd.DataFrame(best_20)
train_smpl_20 = pd.concat([best_20, train_smpl['target']], axis=1)
print(f"train_smpl_20 shape:{train_smpl_20.shape}")

best_50 = SelectKBest(chi2, k=50).fit_transform(train_smpl[train_smpl.columns[:2303]], train_smpl['target'])
best_50 = pd.DataFrame(best_50)
train_smpl_50 = pd.concat([best_50, train_smpl['target']], axis=1)
print(f"train_smpl_50 shape:{train_smpl_50.shape}")

best_100 = SelectKBest(chi2, k=100).fit_transform(train_smpl[train_smpl.columns[:2303]], train_smpl['target'])
best_100 = pd.DataFrame(best_100)
train_smpl_100 = pd.concat([best_100, train_smpl['target']], axis=1)
print(f"train_smpl_100 shape:{train_smpl_100.shape}")


# # Test Train Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_smpl[train_smpl.columns[:2303]], train_smpl['target'], test_size=0.33, random_state=42)

X_train_20, X_test_20, y_train_20, y_test_20 = train_test_split(train_smpl_20[train_smpl_20.columns[:20]], train_smpl_20['target'], test_size=0.33, random_state=42)
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(train_smpl_50[train_smpl_50.columns[:50]], train_smpl_50['target'], test_size=0.33, random_state=42)
X_train_100, X_test_100, y_train_100, y_test_100 = train_test_split(train_smpl_100[train_smpl_100.columns[:100]], train_smpl_100['target'], test_size=0.33, random_state=42)


X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(train_smpl_0[train_smpl_0.columns[:2303]], train_smpl_0['target'], test_size=0.33, random_state=42)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(train_smpl_1[train_smpl_1.columns[:2303]], train_smpl_1['target'], test_size=0.33, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(train_smpl_2[train_smpl_2.columns[:2303]], train_smpl_2['target'], test_size=0.33, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(train_smpl_3[train_smpl_3.columns[:2303]], train_smpl_3['target'], test_size=0.33, random_state=42)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(train_smpl_4[train_smpl_4.columns[:2303]], train_smpl_4['target'], test_size=0.33, random_state=42)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(train_smpl_5[train_smpl_5.columns[:2303]], train_smpl_5['target'], test_size=0.33, random_state=42)
X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(train_smpl_6[train_smpl_6.columns[:2303]], train_smpl_6['target'], test_size=0.33, random_state=42)
X_train_7, X_test_7, y_train_7, y_test_7 = train_test_split(train_smpl_7[train_smpl_7.columns[:2303]], train_smpl_7['target'], test_size=0.33, random_state=42)
X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(train_smpl_8[train_smpl_8.columns[:2303]], train_smpl_8['target'], test_size=0.33, random_state=42)
X_train_9, X_test_9, y_train_9, y_test_9 = train_test_split(train_smpl_9[train_smpl_9.columns[:2303]], train_smpl_9['target'], test_size=0.33, random_state=42)

X_train_10classes = [X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, X_train_7, X_train_8, X_train_9]
X_test_10classes = [X_test_0, X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8, X_test_9]
y_train_10classes = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9]
y_test_10classes = [y_test_0, y_test_1, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7, y_test_8, y_test_9]


# # Modeling

# ## Multinomial Naive Bayes model (multi-class classifier)

# ### Before Features/Attributes Selection

# In[3]:


# In regards to problem 4: Because we are not using Weka, we did not need to apply any
# filters to the data before running Naive Bayes


# In[1]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:


# from sklearn.metrics import confusion_matrix

# y_pred = clf.predict(X_test)
# conf_mat = confusion_matrix(y_test, y_pred)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

#Code accessed from scikit learn
def plot_confusion_matrix(cm,
						  target_names,
						  title='Confusion matrix',
						  cmap=None,
						  normalize=True):

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


# ### 5. For each of the 10 train_smpl_label files, record the first 10 fields, in 
# order of the absolute correlation value for each street sign.

# In[ ]:

from scipy import stats
fileList = [train_smpl_0, train_smpl_1,train_smpl_2, train_smpl_3, train_smpl_4,
			train_smpl_5, train_smpl_6, train_smpl_7, train_smpl_8, train_smpl_9]
for f, file in enumerate(fileList):
	corrArr = []
	for j in file.columns[:-1]:
		corrVal, pVal = stats.pearsonr(file.iloc[:,int(j)], file.iloc[:, -1])
		corrArr.append(corrVal**2)
	print(f"File {f}:")
	print(sorted([(x,i) for (i,x) in enumerate(corrArr)], reverse=True )[:10])


# ### After Features/Attributes Selection (20/50/100)

# In[ ]:

from sklearn.naive_bayes import GaussianNB

X_trains_reduced = [  X_train_20, X_train_50, X_train_100]
X_tests_reduced = [ X_test_20, X_test_50, X_test_100]
y_trains_reduced = [ y_train_20, y_train_50, y_train_100]
y_tests_reduced = [ y_test_20, y_test_50, y_test_100]

for X_trains, X_tests, y_trains, y_tests in zip(X_trains_reduced, X_tests_reduced, y_trains_reduced, y_tests_reduced):    
	clf = GaussianNB()
	clf.fit(X_trains, y_trains)
	print(f"Gaussian NB 20/50/100: {clf.score(X_tests, y_tests)}")

for X_trains, X_tests, y_trains, y_tests in zip(X_trains_reduced, X_tests_reduced, y_trains_reduced, y_tests_reduced):    
	clf = MultinomialNB()
	clf.fit(X_trains, y_trains)
	print(f"Multinomial NB 20/50/100: {clf.score(X_tests, y_tests)}")

# ## Gaussian Naive Bayes model (mono-class classifier)

# In[ ]:



scores = []
for X_train, X_test, y_train, y_test in zip(X_train_10classes, X_test_10classes, y_train_10classes, y_test_10classes):    
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	scores.append(clf.score(X_test, y_test))
	print(f"Gaussian NB 10 classes:{clf.score(X_test, y_test)}")
print("Gaussian NB Mean: ", sum(scores)/len(scores))

scores2 = []
for X_train, X_test, y_train, y_test in zip(X_train_10classes, X_test_10classes, y_train_10classes, y_test_10classes):    
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	scores2.append(clf.score(X_test, y_test))
	print(f"Multinomial NB 10 classes:{clf.score(X_test, y_test)}")
print("Multinomial NB Mean: ", sum(scores2)/len(scores2))


# In[ ]:


#Multinomial Nb is less accurate than Gaussian Nb ( 0.45 <= 0.69 )


# # Analysis for 4 & 7

# ### 4. Explain the reason for choosing and using these filters. Once you can run the algorithm, record, compare and analyse the classifier’s accuracy on different classes (as given by the Weka Summary and the confusion matrix).

# #### Since we are not using Weka, we did not need to apply any filters to the data before running Naive Bayes.

# ### 7. What kind of information about this data set did you learn, as a result of the above experiments? You should ask questions such as: Which streets signs are harder to recognise? Which street signs are most easily confused? Which attributes (fields) are more reliable and which are less reliable in classification of street signs? What was the purpose of Tasks 5 and 6? What would happen if the data sets you used in Tasks 4, 5 and 6 were not randomised? What would happen if there is cross-correlation between the non-class attributes? You will get more marks for more interesting and ``out of the box” questions and answers. Explain your conclusions logically and formally, using the material from the lecture notes and from your own reading to interpret the results that Weka produces.

# #### Analysis....

# In[ ]:




