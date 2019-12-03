#Author: Ateeth Kumar Thirukkovulur - Please give original credits when reusing code
#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd  
import seaborn as sns
import string 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 

punctuation_signs = list("!#$%&'()*+,-./:;<=>?@[\]^_`{|}~") 
wordnet_lemmatizer = WordNetLemmatizer() 
lemmatized_text_list = [] 
stop_words = list(stopwords.words('english')) 
# nltk.download('punkt') 
# nltk.download('stopwords') 
# nltk.download('wordnet') 

## Creating a pandas dataframe by importing classification dataset as type string - import item description, item class id and item category name 
df = pd.read_csv("Classification_Dataset.csv", dtype=str, keep_default_na=False) 

## Calculating the rows of the imported dataframe
nrows = len(df) 

## Analysis of text
df['char_count'] = df['item_desc'].apply(len)
df['word_count'] = df['item_desc'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['item_desc'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['title_word_count'] = df['item_desc'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['item_desc'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

## Text cleaning of item description - Replace special characters, punctuations, numbers and stop words. 
## Also applying lemmatizing text
df['Content_Parsed'] = df['item_desc'].str.replace("\r", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("\n", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace("\t", " ")
df['Content_Parsed'] = df['Content_Parsed'].str.replace('"', '')
df['Content_Parsed'] = df['Content_Parsed'].str.lower()
for punct_sign in punctuation_signs:
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(punct_sign, '')t
df['Content_Parsed']=df['Content_Parsed'].str.replace('\d+', '')
for row in range(0, nrows):
    ## Create an empty list containing lemmatized words
    lemmatized_list = []
    ## Save the text and its words into an object
    text = df.loc[row]['Content_Parsed']
    text_words = text.split(" ")
    ## Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    ## Join the list
    lemmatized_text = " ".join(lemmatized_list)
    ## Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)
df['Content_Parsed'] = lemmatized_text_list

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed'] = df['Content_Parsed'].str.replace(regex_stopword, '')

## Assigning labels for all distinct categories into a python dictionary datatype
s = df.groupby("classification_name").item_desc.agg(lambda x:len(x.unique()))
category_codes={}
for i in range(len(s)):
    category_codes[s.index[i]]=i
    
df['Category_Code'] = df['classification_name']
df = df.replace({'Category_Code':category_codes})

## Importing sklearn modules that are to be used and tested to fit and predict training and test data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## Splitting given dataframe into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'], 
                                                    df['Category_Code'], 
                                                    test_size=0.6, 
                                                    random_state=12)

## Converting training and testing data into tf - idf vectors to pass to model for machine learning
ngram_range = (1,2)
min_df = 10
max_df = 1.0
max_features = 300

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test

##---------------------Trials by looking at features names of given categories using unigrams and bigrams
# from sklearn.feature_selection import chi2
# import numpy as np

# for Product, category_id in sorted(category_codes.items()):
#     features_chi2 = chi2(features_train, labels_train == category_id)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(tfidf.get_feature_names())[indices]
#    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#     print("# '{}' category:".format(Product))
#     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
#     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
#     print("")


##----------------------Base model to tune
svc = svm.SVC(C=1, gamma=10, kernel='poly', degree=4, random_state=42, probability=True)

##----------------------Uncomment block for applying random search grid to find best parameters
# C = [.001, .01]
# gamma = [.01, .1, 1, 10, 100]
# degree = [1, 2, 3, 4, 5]
# kernel = ['linear', 'rbf', 'poly']
# probability = [True]

# Create the random grid
# random_grid = {'C': C,
#               'kernel': kernel,
#               'gamma': gamma,
#               'degree': degree,
#               'probability': probability
#              }

# random_search = RandomizedSearchCV(estimator=svc,
#                                    param_distributions=random_grid,
#                                    n_iter=10,
#                                    scoring='accuracy',
#                                    cv=3, 
#                                    verbose=1, 
#                                    random_state=12)
##----------------------End of Uncomment block for applying random search grid to find best parameters

##----------------------Uncomment block for using multiclass learning using output-codes
occ = OutputCodeClassifier(svc,code_size=2, random_state=8)
##----------------------End of Uncomment block for using multiclass learning using output-codes

## Fit your chosen model by changing the vaiable before the period to either - svc, random_search, grid_search or occ 
occ.fit(features_train, labels_train)

##----------------------Uncomment required block for finding out best parameters if using the random search or grid search for best accuracy
# print("The best hyperparameters from Random Search are:")
# print(random_search.best_params_)
# print("")
# print("The mean accuracy of a model with these hyperparameters is:")
# print(random_search.best_score_)
##----------------------End of Uncomment block for finding out best parameters if using the random search for best accuracy

def get_key(val):
    identifiedKey = [k for k,v in category_codes.items() if v == val]
    if  len(identifiedKey) == 0:
        return "No value"
    return identifiedKey[0]

## Actual code to run for large data prediction
#labels_predict = svc.predict(features_test)
#print(accuracy_score(labels_test_test, labels_predict_test))

## Test predict only for 2000 rows due to machine constraints
very_small_sample_size = 2000
labels_test_test = labels_test.head(very_small_sample_size)
input_test = X_test.head(very_small_sample_size)
features_test_input = tfidf.transform(input_test).toarray()
labels_predict_test = occ.predict(features_test_input)

## Print the test results accuracy compared to actual
print("The training accuracy is: ")
print(accuracy_score(labels_test_test, labels_predict_test))

print("Classification report is as follows: ")
print(classification_report(labels_test_test, labels_predict_test))

## Uncomment below for printing confusion matrix
# cm=confusion_matrix(labels_test_test, labels_predict_test)
# sns.heatmap(cm, annot=True)

## For very small given sample size print - Item Desc || Actual test category || Predicted test category
for i in range(very_small_sample_size):
    print (input_test.values[i], " || ", get_key(labels_test.values[i]), " || ", get_key(labels_predict_test[i]))

