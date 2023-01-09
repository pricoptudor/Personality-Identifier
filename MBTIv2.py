import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import pickle
import os
import re
import tkinter as tk
from tkinter import filedialog
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC # LinearSVC is approximate SVC but more efficient with large dataset
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

### One classifier with 16 classes => 43.93% accuracy train and 10.42% accuracy test
### => 4 classifiers: I-E || N-S || T-F || J-P

class Dataset:
    def read_dataset(self, dataset_file='mbti_dataset.csv'):
        """Dataset has no missing values

        Shape: (8675, 2) -> type | posts
        Each row: 50 posts split by '|||'
        16 personality types: I-E, N-S, T-F, J-P
        """
        self.dataset = pd.read_csv(dataset_file)
        self.model = Model()
        self.lemmatizer = WordNetLemmatizer()
        self.Tfidf = TfidfVectorizer(max_features=1000).fit(self.dataset['posts'])
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.url_regex = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

    def clean_text(self):
        df_length = []
        cleaned_text = []
        str_punc = string.punctuation
        for text in self.dataset.posts:
            text = text.lower()
            text = re.sub('http.*?([ ]|\|\|\||$)', ' ', text)
            text = re.sub(self.url_regex, ' ', text)
            text = re.sub('['+re.escape(str_punc)+']'," ",  text)
            text = re.sub('(\[|\()*\d+(\]|\))*', ' ', text)

            # Remove string marks
            text = re.sub('[’‘“\.”…–]', '', text)
            text = re.sub('[^(\w|\s)]', '', text)
            text = re.sub('(gt|lt)', '', text)
            
            df_length.append(len(text.split()))
            cleaned_text.append(text)
        self.dataset.posts = cleaned_text

    def lemmatization(self):
        for i in range(len(self.dataset['posts'])):
            self.dataset['posts'][i] = nltk.word_tokenize(self.dataset['posts'][i])
            self.dataset['posts'][i] = [self.lemmatizer.lemmatize(word) for word in self.dataset['posts'][i]]
            self.dataset['posts'][i] = ' '.join(self.dataset['posts'][i])
        
    def remove_stopwords(self):
        self.dataset['posts'] = self.dataset['posts'].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop_words)]))

    def vectorize(self):
        X = self.Tfidf.transform(self.dataset['posts'])
        y = self.dataset['type']
        return (X, y)

    def preprocess_dataset(self):
        # self.types = np.unique(np.array(self.dataset['type']))
        # self.posts_no_by_type = self.dataset.groupby(['type']).count() * 50
        if os.path.exists('MBTI-instances.pickle') and os.path.exists('MBTI-labels.pickle'):
            X = self.model.load_data('MBTI-instances.pickle')
            y = self.model.load_data('MBTI-labels.pickle')
        else:
            self.clean_text()
            self.lemmatization()
            self.remove_stopwords()
            (X, y) = self.vectorize()

            self.model.save_data('MBTI-instances.pickle', X)
            self.model.save_data('MBTI-labels.pickle', y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    def boolean(self, type, pos, first, second):
        if type[pos] == first:
            return 1
        elif type[pos] == second:
            return 0

class NaiveBayes:
    def __init__(self, dataset):
        self.dataset = dataset

    def build_classifiers(self):
        if os.path.exists('NaiveBayes_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('NaiveBayes_IE')
        else:
            (self.IEClassifier, accuracy) = self.grid_search(0, 'I', 'E')
            self.dataset.model.save_model('NaiveBayes_IE', self.IEClassifier)

        if os.path.exists('NaiveBayes_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('NaiveBayes_NS')
        else:
            (self.NSClassifier, accuracy) = self.grid_search(1, 'N', 'S')
            self.dataset.model.save_model('NaiveBayes_NS', self.NSClassifier)

        if os.path.exists('NaiveBayes_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('NaiveBayes_TF')
        else:
            (self.TFClassifier, accuracy) = self.grid_search(2, 'T', 'F')
            self.dataset.model.save_model('NaiveBayes_TF', self.TFClassifier)

        if os.path.exists('NaiveBayes_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('NaiveBayes_JP')
        else:
            (self.JPClassifier, accuracy) = self.grid_search(3, 'J', 'P')
            self.dataset.model.save_model('NaiveBayes_JP', self.JPClassifier)

    def grid_search(self, classifier_no, first_label, second_label):
        pipeline = Pipeline([('nb', BernoulliNB())])

        param_grid = {
            'nb__alpha': [0.1, 1.0, 10]
        }

        # Create a GridSearchCV object with 5-fold cross-validation (i do not have to split data into train/test)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)

        # Fit the grid search object to the training data
        labels = [self.dataset.boolean(type, classifier_no, first_label, second_label) for type in self.dataset.y_train]
        grid_search.fit(self.dataset.X_train, labels)

        # Print the best parameters and the best score
        print('Stats:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return (grid_search.best_estimator_, grid_search.best_score_*100)

    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class SVM:
    def __init__(self, dataset):
        self.dataset = dataset

    def build_classifiers(self):
        if os.path.exists('SVM_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('SVM_IE')
        else:
            (self.IEClassifier, accuracy) = self.grid_search(0, 'I', 'E')
            self.dataset.model.save_model('SVM_IE', self.IEClassifier)

        if os.path.exists('SVM_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('SVM_NS')
        else:
            (self.NSClassifier, accuracy) = self.grid_search(1, 'N', 'S')
            self.dataset.model.save_model('SVM_NS', self.NSClassifier)

        if os.path.exists('SVM_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('SVM_TF')
        else:
            (self.TFClassifier, accuracy) = self.grid_search(2, 'T', 'F')
            self.dataset.model.save_model('SVM_TF', self.TFClassifier)

        if os.path.exists('SVM_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('SVM_JP')
        else:
            (self.JPClassifier, accuracy) = self.grid_search(3, 'J', 'P')
            self.dataset.model.save_model('SVM_JP', self.JPClassifier)

    def grid_search(self, classifier_no, first_label, second_label):
        pipeline = Pipeline([('svm', SVC())])

        param_grid = {
            'svm__kernel': ['linear', 'rbf'],
            'svm__C': [0.1, 1, 10],
            'svm__gamma': [0.01, 0.1, 1],
            'svm__max_iter': [10000, 100000]
        }

        # Create a GridSearchCV object with 5-fold cross-validation (i do not have to split data into train/test)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)

        # Fit the grid search object to the training data
        labels = [self.dataset.boolean(type, classifier_no, first_label, second_label) for type in self.dataset.y_train]
        grid_search.fit(self.dataset.X_train, labels)

        # Print the best parameters and the best score
        print('Stats:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return (grid_search.best_estimator_, grid_search.best_score_*100)

    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class KNN:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def build_classifiers(self):
        if os.path.exists('KNN_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('KNN_IE')
        else:
            (self.IEClassifier, accuracy) = self.grid_search(0, 'I', 'E')
            self.dataset.model.save_model('KNN_IE', self.IEClassifier)

        if os.path.exists('KNN_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('KNN_NS')
        else:
            (self.NSClassifier, accuracy) = self.grid_search(1, 'N', 'S')
            self.dataset.model.save_model('KNN_NS', self.NSClassifier)

        if os.path.exists('KNN_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('KNN_TF')
        else:
            (self.TFClassifier, accuracy) = self.grid_search(2, 'T', 'F')
            self.dataset.model.save_model('KNN_TF', self.TFClassifier)

        if os.path.exists('KNN_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('KNN_JP')
        else:
            (self.JPClassifier, accuracy) = self.grid_search(3, 'J', 'P')
            self.dataset.model.save_model('KNN_JP', self.JPClassifier)

    def grid_search(self, classifier_no, first_label, second_label):
        pipeline = Pipeline([('knn', KNeighborsClassifier())])

        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15]
        }

        # Create a GridSearchCV object with 5-fold cross-validation (i do not have to split data into train/test)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)

        # Fit the grid search object to the training data
        labels = [self.dataset.boolean(type, classifier_no, first_label, second_label) for type in self.dataset.y_train]
        grid_search.fit(self.dataset.X_train, labels)

        # Print the best parameters and the best score
        print('Stats:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return (grid_search.best_estimator_, grid_search.best_score_*100)

    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class LogReg:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def build_classifiers(self):
        if os.path.exists('LogReg_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('LogReg_IE')
        else:
            (self.IEClassifier, accuracy) = self.grid_search(0, 'I', 'E')
            self.dataset.model.save_model('LogReg_IE', self.IEClassifier)

        if os.path.exists('LogReg_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('LogReg_NS')
        else:
            (self.NSClassifier, accuracy) = self.grid_search(1, 'N', 'S')
            self.dataset.model.save_model('LogReg_NS', self.NSClassifier)

        if os.path.exists('LogReg_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('LogReg_TF')
        else:
            (self.TFClassifier, accuracy) = self.grid_search(2, 'T', 'F')
            self.dataset.model.save_model('LogReg_TF', self.TFClassifier)

        if os.path.exists('LogReg_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('LogReg_JP')
        else:
            (self.JPClassifier, accuracy) = self.grid_search(3, 'J', 'P')
            self.dataset.model.save_model('LogReg_JP', self.JPClassifier)

    def grid_search(self, classifier_no, first_label, second_label):
        pipeline = Pipeline([('logreg', LogisticRegression())])

        param_grid = {
            'logreg__C': [0.1, 1.0, 10.0],
            'logreg__penalty': ['l1', 'l2']
        }

        # Create a GridSearchCV object with 5-fold cross-validation (i do not have to split data into train/test)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)

        # Fit the grid search object to the training data
        labels = [self.dataset.boolean(type, classifier_no, first_label, second_label) for type in self.dataset.y_train]
        grid_search.fit(self.dataset.X_train, labels)

        # Print the best parameters and the best score
        print('Stats:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return (grid_search.best_estimator_, grid_search.best_score_*100)

    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class RandomForest:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def build_classifiers(self):
        if os.path.exists('RandomForest_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('RandomForest_IE')
        else:
            (self.IEClassifier, accuracy) = self.grid_search(0, 'I', 'E')
            self.dataset.model.save_model('RandomForest_IE', self.IEClassifier)

        if os.path.exists('RandomForest_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('RandomForest_NS')
        else:
            (self.NSClassifier, accuracy) = self.grid_search(1, 'N', 'S')
            self.dataset.model.save_model('RandomForest_NS', self.NSClassifier)

        if os.path.exists('RandomForest_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('RandomForest_TF')
        else:
            (self.TFClassifier, accuracy) = self.grid_search(2, 'T', 'F')
            self.dataset.model.save_model('RandomForest_TF', self.TFClassifier)

        if os.path.exists('RandomForest_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('RandomForest_JP')
        else:
            (self.JPClassifier, accuracy) = self.grid_search(3, 'J', 'P')
            self.dataset.model.save_model('RandomForest_JP', self.JPClassifier)

    def grid_search(self, classifier_no, first_label, second_label):
        pipeline = Pipeline([('randomforest', RandomForestClassifier())])

        param_grid = {
            'randomforest__n_estimators': [10, 50, 100],
            'randomforest__max_depth': [None, 5, 10],
            'randomforest__min_samples_split': [2, 5, 10],
            'randomforest__min_samples_leaf': [1, 2, 4]
        }

        # Create a GridSearchCV object with 5-fold cross-validation (i do not have to split data into train/test)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=10)

        # Fit the grid search object to the training data
        labels = [self.dataset.boolean(type, classifier_no, first_label, second_label) for type in self.dataset.y_train]
        grid_search.fit(self.dataset.X_train, labels)

        # Print the best parameters and the best score
        print('Stats:')
        print(grid_search.best_params_)
        print(grid_search.best_score_)

        return (grid_search.best_estimator_, grid_search.best_score_*100)

    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class Stacker:
    def __init__(self, dataset):
        self.dataset = dataset

    def combine_models(self, first_model, second_model, third_model, fourth_model, fifth_model):
        if os.path.exists('StackedModel_IE.pickle'):
            self.IEClassifier = self.dataset.model.load_model('StackedModel_IE')
        else:
            self.IEClassifier = StackingClassifier(
                estimators=[('1', first_model.IEClassifier), 
                            ('2', second_model.IEClassifier), 
                            ('3', third_model.IEClassifier),
                            ('4', fourth_model.IEClassifier),
                            ('5', fifth_model.IEClassifier)],
                final_estimator=LogisticRegression(),
                verbose=10
            )
            labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_train]
            self.IEClassifier.fit(self.dataset.X_train, labels)
            self.dataset.model.save_model('StackedModel_IE', self.IEClassifier)

        if os.path.exists('StackedModel_NS.pickle'):
            self.NSClassifier = self.dataset.model.load_model('StackedModel_NS')
        else:
            self.NSClassifier = StackingClassifier(
                estimators=[('1', first_model.NSClassifier), 
                            ('2', second_model.NSClassifier), 
                            ('3', third_model.NSClassifier),
                            ('4', fourth_model.NSClassifier),
                            ('5', fifth_model.NSClassifier)],
                final_estimator=LogisticRegression(),
                verbose=10
            )
            labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_train]
            self.NSClassifier.fit(self.dataset.X_train, labels)
            self.dataset.model.save_model('StackedModel_NS', self.NSClassifier)

        if os.path.exists('StackedModel_TF.pickle'):
            self.TFClassifier = self.dataset.model.load_model('StackedModel_TF')
        else:
            self.TFClassifier = StackingClassifier(
                estimators=[('1', first_model.TFClassifier), 
                            ('2', second_model.TFClassifier), 
                            ('3', third_model.TFClassifier),
                            ('4', fourth_model.TFClassifier),
                            ('5', fifth_model.TFClassifier)],
                final_estimator=LogisticRegression(),
                verbose=10
            )
            labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_train]
            self.TFClassifier.fit(self.dataset.X_train, labels)
            self.dataset.model.save_model('StackedModel_TF', self.TFClassifier)

        if os.path.exists('StackedModel_JP.pickle'):
            self.JPClassifier = self.dataset.model.load_model('StackedModel_JP')
        else:
            self.JPClassifier = StackingClassifier(
                estimators=[('1', first_model.JPClassifier), 
                            ('2', second_model.JPClassifier), 
                            ('3', third_model.JPClassifier),
                            ('4', fourth_model.JPClassifier),
                            ('5', fifth_model.JPClassifier)],
                final_estimator=LogisticRegression(),
                verbose=10
            )
            labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_train]
            self.JPClassifier.fit(self.dataset.X_train, labels)
            self.dataset.model.save_model('StackedModel_JP', self.JPClassifier)

    
    def test_accuracy(self):
        labels = [self.dataset.boolean(type, 0, 'I', 'E') for type in self.dataset.y_test]
        self.IEAccuracy = self.IEClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 1, 'N', 'S') for type in self.dataset.y_test]
        self.NSAccuracy = self.NSClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 2, 'T', 'F') for type in self.dataset.y_test]
        self.TFAccuracy = self.TFClassifier.score(self.dataset.X_test, labels)

        labels = [self.dataset.boolean(type, 3, 'J', 'P') for type in self.dataset.y_test]
        self.JPAccuracy = self.JPClassifier.score(self.dataset.X_test, labels)

        print(f'Accuracy per model: {self.IEAccuracy}, {self.NSAccuracy}, {self.TFAccuracy}, {self.JPAccuracy}')


class LSTMNeural:
    ## COME BACK HERE
    def __init__(self, dataset):
        self.dataset = dataset

    def build_classifiers(self):
        print('Train shape: ', self.dataset.X_train.shape)
        print('Test shape: ', self.dataset.X_test.shape)
        
        model = Sequential()
        model.add(Embedding(input_dim=self.dataset.X_train.shape[0], output_dim=100, input_length=self.dataset.X_train.shape[1]))
        model.add(LSTM(units=100))
        model.add(Dense(units=1, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.dataset.X_train, self.dataset.y_train, batch_size=1500, epochs=500)

        exit()

class Plot:
    def plot_acc(self, classifier_name, accuracy):
        pass

class Model:
    # Model name example: NaiveBayes_IE (model_category)
    def save_model(self, model_name, classifier):
        f = open(f'{model_name}.pickle', 'wb')
        pickle.dump(classifier, f)
        f.close()

    def load_model(self, model_name):
        f = open(f'{model_name}.pickle', 'rb')
        classifier = pickle.load(f)
        f.close()
        return classifier

    def save_data(self, filename, data):
        f = open(filename, 'wb')
        pickle.dump(data, f)
        f.close()
    
    def load_data(self, filename):
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()
        return data

class MBTI:
    def __init__(self, dataset, IEClassifier, NSClassifier, TFClassifier, JPClassifier):
        self.dataset = dataset
        self.IEClassifier = IEClassifier
        self.NSClassifier = NSClassifier
        self.TFClassifier = TFClassifier
        self.JPClassifier = JPClassifier

    def read_posts(self, filename='posts.txt'):
        file_input = open(filename, encoding='utf8')
        posts = file_input.readlines()
        posts = posts[0].split('|||')
        self.posts = pd.DataFrame(posts, columns=['posts'])
        
    def read_post_string(self, data):
        posts = data.split('|||')
        self.posts = pd.DataFrame(posts, columns=['posts'])

    def preprocess_input(self):
        self.posts['posts'] = self.posts['posts'].apply(self.clean_text)
        self.lemmatization()
        self.remove_stopwords()
        X=self.vectorize()
        return X ## TODO: take X and make predictions in order to get accuracy

    def clean_text(self, text):
        text = text.lower()
    
        text = re.sub("@[A-Za-z0-9_]+","", text)
        text = re.sub("#[A-Za-z0-9_]+","", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www.\S+", "", text)
        text = re.sub('[()!?]', ' ', text)
        text = re.sub('\[.*?\]',' ', text)
        text = re.sub("[^a-z0-9]"," ", text)
        
        return text

    def lemmatization(self):
        for i in range(len(self.posts['posts'])):
            self.posts['posts'][i] = nltk.word_tokenize(self.posts['posts'][i])
            self.posts['posts'][i] = [self.dataset.lemmatizer.lemmatize(word) for word in self.posts['posts'][i]]
            self.posts['posts'][i] = ' '.join(self.posts['posts'][i])
        
    def remove_stopwords(self):
        self.posts['posts'] = self.posts['posts'].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.dataset.stop_words)]))

    def vectorize(self):
        X = self.dataset.Tfidf.transform(self.posts['posts'])
        return X

    def predict(self, input):
        IE = self.IEClassifier.predict(input)[0]
        NS = self.NSClassifier.predict(input)[0]
        TF = self.TFClassifier.predict(input)[0]
        JP = self.JPClassifier.predict(input)[0]

        result = ''
        if(IE == 1):
            result+='I'
        if(IE == 0):
            result+='E'
        if(NS == 1):
            result+='N'
        if(NS == 0):
            result+='S'
        if(TF == 1):
            result+='T'
        if(TF == 0):
            result+='F'
        if(JP == 1):
            result+='J'
        if(JP == 0):
            result+='P'
        return result

    def get_results(self, posts, model_name, user_name, traasits=[]):
        a = []
        trait1 = pd.DataFrame([0,0,0,0],['I','N','T','J'],['count'])
        trait2 = pd.DataFrame([0,0,0,0],['E','S','F','P'],['count'])
        for post in posts:
            a += [self.predict(post)]
        for i in a:
            for j in ['I','N','T','J']:
                if j in i:
                    trait1.loc[j] += 1                
            for j in ['E','S','F','P']:
                if j in i:
                    trait2.loc[j] += 1 
        trait1 = trait1.T
        trait1 = trait1*100/posts.shape[0]
        trait2 = trait2.T
        trait2 = trait2*100/posts.shape[0]
        
        
        #Finding the personality
        YourTrait = ''
        for i,j in zip(trait1,trait2):
            temp = max(trait1[i][0],trait2[j][0])
            if(trait1[i][0]==temp):
                YourTrait += i  
            if(trait2[j][0]==temp):
                YourTrait += j
        traasits +=[YourTrait] 
        
        #Plotting
        labels = np.array(['I-E','N-S','T-F','J-P'])

        intj = trait1.loc['count']
        ind = np.arange(4)
        width = 0.4
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, intj, width, color='royalblue')

        esfp = trait2.loc['count']
        rects2 = ax.bar(ind+width, esfp, width, color='seagreen')

        fig.set_size_inches(10, 7)

        ax.set_xlabel('Finding the MBTI Trait', size = 18)
        ax.set_ylabel('Trait Percent (%)', size = 18)
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(0,105, step= 10))
        ax.set_title('Your Personality is '+YourTrait,size = 20)
        plt.grid(True)
        
        fig.savefig(f'{model_name}-{user_name}.png', dpi=200)
        
        plt.show()
        return traasits

if __name__ == '__main__':
    dataset = Dataset()
    dataset.read_dataset()
    dataset.preprocess_dataset()

    naive_bayes = NaiveBayes(dataset)
    naive_bayes.build_classifiers()
    # naive_bayes.test_accuracy()

    svm = SVM(dataset)
    svm.build_classifiers()
    # svm.test_accuracy()

    knn = KNN(dataset)
    knn.build_classifiers()
    # knn.test_accuracy()

    logreg = LogReg(dataset)
    logreg.build_classifiers()
    # logreg.test_accuracy()

    randomforest = RandomForest(dataset)
    randomforest.build_classifiers()
    # randomforest.test_accuracy()

    stack = Stacker(dataset)
    stack.combine_models(naive_bayes, svm, knn, logreg, randomforest)
    # stack.test_accuracy()

    lstm = LSTMNeural(dataset)
    lstm.build_classifiers()

    plot = Plot()
    classifier = stack # default classifier in case none chosen
    file_content = '' # default file content for file button
    text_content = ''

    # mbti = MBTI(dataset, 
    #             stack.IEClassifier,
    #             stack.NSClassifier,
    #             stack.TFClassifier,
    #             stack.JPClassifier)

    # mbti.read_posts()
    # posts = mbti.preprocess_input()
    # trait = mbti.get_results(posts, 'Model', 'Tudor')

    window = tk.Tk()
    window.geometry("940x600")

    def svm_classifier():
        global classifier
        global svm
        print('SVM')
        classifier=svm


    def rf_classifier():
        global classifier
        global randomforest
        print('Random forest')
        classifier=randomforest

    def logreg_classifier():
        global classifier
        global logreg
        print('Logreg')
        classifier=logreg

    def nb_classifier():
        global classifier
        global naive_bayes
        print('naive bayes')
        classifier=naive_bayes

    def knn_classifier():
        global classifier
        global knn
        print('KNN')
        classifier=knn

    def stack_classifier():
        global classifier
        global stack
        print('stacked')
        classifier=stack

    def load_file():
        file_path = filedialog.askopenfilename()
        with open(file_path, "r", encoding='utf8') as file:
            file_contents = file.readlines()[0]

        global file_content
        file_content = file_contents

    def compute():
        global classifier
        global file_content
        global text_content
        global dataset
        global textbox1

        text_content = textbox1.get("1.0", tk.END)

        mbti = MBTI(dataset, 
                classifier.IEClassifier,
                classifier.NSClassifier,
                classifier.TFClassifier,
                classifier.JPClassifier)
        if text_content.strip() != '':
            # print(f'::{text_content}::')
            # print('Text chosen')
            mbti.read_post_string(text_content)
        elif file_content.strip() != '':
            # print('File chosen')
            mbti.read_post_string(file_content)
        else:
            print('My guy insert text or file!!')

        print('Computing your personality...')
        # print('Text: ', text_content)
        # print('File: ', file_content)
        posts = mbti.preprocess_input()
        trait = mbti.get_results(posts, 'Model', 'Tudor')

    button1 = tk.Button(text="SVM", height=5, width=21, command=svm_classifier)
    button2 = tk.Button(text="Random Forest", height=5, width=21, command=rf_classifier)
    button3 = tk.Button(text="Logistic Regression", height=5, width=21, command=logreg_classifier)
    button4 = tk.Button(text="Naive Bayes", height=5, width=21, command=nb_classifier)
    button5 = tk.Button(text="KNN", height=5, width=21, command=knn_classifier)
    button6 = tk.Button(text="Stacked", height=5, width=21, command=stack_classifier)
    textbox1 = tk.Text(wrap=tk.WORD)
    textbox1.insert(tk.END, "Input posts separated by '||' and find your personality...")
    button7 = tk.Button(text="Load file", command=load_file)
    button8 = tk.Button(text="Compute personality", command=compute)

    button1.grid(row=0, column=0)
    button2.grid(row=0, column=1)
    button3.grid(row=0, column=2)
    button4.grid(row=0, column=3)
    button5.grid(row=0, column=4)
    button6.grid(row=0, column=5)
    textbox1.place(x=10, y=100, width=920, height=300)
    button7.place(x=365, y=450, width=200, height=100)
    button8.place(x=700, y=450, width=200, height=100)

    textbox1.bind('<FocusIn>', lambda e: textbox1.delete("1.0", tk.END))

    window.mainloop()
