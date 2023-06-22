# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
import string
import warnings
warnings.filterwarnings('ignore')

# Load the review file
data = pd.read_csv('reviews.csv', delimiter='\t')
data.head()

# Check the RatingValue column
print(data['RatingValue'])

# Data preprossing
# Transfer RatingValue to the sentiment list
Sentiment = []

for dataset in data['RatingValue']:
    if dataset == 1 or dataset == 2:
      Sentiment.append(0)
    elif dataset == 3:
      Sentiment.append(1) 
    else:
      Sentiment.append(2)

data['Sentiment'] = Sentiment
print(data['Sentiment'])

# Drop positive ratings in order to balance data 
df = data[data['Sentiment'] == 2]
df = df.sample(frac=0.2, random_state=33)

# Concat the 2 rating with the rest of table 
data = pd.concat([df,data[data['Sentiment'] != 2]])
data['Sentiment'].value_counts()

# Drop the unneeded columns and switch the order of columns around
data.drop(columns='Name', inplace = True)
data.drop(columns='RatingValue', inplace = True)
data.drop(columns='DatePublished', inplace = True)
data = data[['Sentiment', 'Review']]
data.head()

# Partition data into train, validation sets
x = data
y = data['Sentiment']
x_train, x_validate, y_train, y_validate = train_test_split(x,y,test_size=0.3,random_state=21)

# Save the traing file
x_train.to_csv('train.csv', index=False)

# Save the valid file
x_validate.to_csv('valid.csv', index=False)

# Load the train data
train = pd.read_csv('train.csv')

train.head()

# Use bag-of-words to train and transform data from text to numbers
vectorizer = CountVectorizer()
train = vectorizer.fit_transform(train['Review'])

# Training using MultinomialNB
multiply_nb = MultinomialNB() 
multiply_nb.fit(train, y_train)

# Traning using decision tree
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(train, y_train)

# Traning using Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train, y_train)

# Load the valid data
valid = pd.read_csv('valid.csv')

# Accuracy for MultinomialNB
valid = vectorizer.transform(valid['Review'])
print("accuracy for MultinomialNB: ", multiply_nb.score(valid,y_validate))

# Accuracy for decision tree
print("accuracy for decision tree: ", decision_tree.score(valid,y_validate))

# Accuracy for Logistic Regression
print("accuracy for Logistic Regression: ", log_reg.score(valid,y_validate))

print("For model selection, MultinomialNB has the highest score, then we will use MultinomialNB to train the model")

print("accuracy on the test set:", multiply_nb.score(valid,y_validate))

predction = multiply_nb.predict(valid)
print("Average f1-score on the test set: ", f1_score(y_validate, predction,average='macro'))

predction = multiply_nb.predict(valid)
score = f1_score(y_validate, predction,average=None)
print("Class-wise F1 scores:")
print("negative: ",score[0])
print("neutral: ",score[1])
print("postive: ",score[2])

# create the confusion matrix
confusion = confusion_matrix(y_validate, predction)

# normalize the confusion matrix
normalized_confusion = confusion / confusion.astype(np.float).sum(axis=1)

# lable the columns and rows
col_name = ['predicted negative', 'predicted neutral', 'predicted positive']
row_name = ['actual negative', 'actual neutral', 'actual positive']
normalized_confusion = pd.DataFrame(normalized_confusion, index = row_name, columns = col_name)
print("Confusion matrix:\n{}".format(normalized_confusion))