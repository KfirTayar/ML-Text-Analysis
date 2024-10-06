import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------
from sklearn import preprocessing
from sklearn import preprocessing, metrics

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# ---------------------------------------

#--------- Text analysis and Hebrew text analysis imports:
#vectorizers:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#regular expressions:
import re
# --------------------------------------

train_filename = 'annotated_corpus_for_train.csv'
test_filename  = 'corpus_for_test.csv'
df_train = pd.read_csv(train_filename, index_col=None, encoding='utf-8')
df_test  = pd.read_csv(test_filename, index_col=None, encoding='utf-8')

st.subheader("Train Data Preview")
st.write(df_train.head())
st.write("Train DF Shape:", df_train.shape)

st.subheader("Test Data Preview")
st.write(df_test.head())
st.write("Test DF Shape:", df_test.shape)

st.subheader("Train Df describe")
st.write(df_train.describe())

st.title("Data Pre-Processing")

# Clean the story and split him to tokens
def makeTokenization(story):
    
    story = story.strip(',:;{}*().- ')
    story = story.replace("'", "").replace(".", "").replace("\\", "").replace(",", "").replace(";", "").replace("!","").replace("?","").replace("(","").replace(")","").replace("*", "").replace("/", "")
    story = re.sub(r'\d+', '', story)
    story = re.sub(r'[a-z]+ | [A-Z]+', '', story)
    story = story.split(' ')
        
    return story

# Testing the tokenization function
st.subheader("Tokenization")
a = "123123טסט טסט טסט ;;;"
st.write("Test Tokenization function on this text:", a)
st.write("Output:", makeTokenization("טסט טסט טסט123123 ;;;"))

# Make the gender col to numeric values
df_train['gender'] = df_train['gender'].map({'f':0, 'm':1})

st.title("Vectorization")

st.subheader("Trainset Vectorization")
# Create the vectorizer
vec = TfidfVectorizer(tokenizer=makeTokenization, max_features=3000, min_df=2, max_df=0.95, ngram_range=(1,2))
X_train = vec.fit_transform(df_train.story)
y_train = df_train.gender

# Normalize
X_train = preprocessing.normalize(X_train, norm='l2')
st.write(X_train[:3])

st.subheader("Testset Vectorization")
X_test = vec.transform(df_test.story)

# Normalize
X_test = preprocessing.normalize(X_test, norm='l2')
st.write(X_test[:3])

st.title("Model Selection & Model Evaluation")

st.subheader("Cross Validation & GridSearchCV")

# Make the y_pred ndarray look more user friendly
def format_yPredValues(arr):
    
    # Convert the array to a list if it's a numpy array
    arr_li = arr.tolist()
    
    # Create the color formatted array for Streamlit
    color_arr = [
        f'<span style="color: magenta; font-weight: bold;"> f </span>' if i == 0 else 
        f'<span style="color: green; font-weight: bold;"> m </span>' for i in arr_li
    ]
    
    # Join the colored strings into one string
    color_string = "".join(color_arr)
    
    return color_string

# Train the model we chose to check
def train_model(model, X_train, y_train):
    if(model == 'KNN'):
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        
    if(model == 'DT'):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
    if(model == 'MLP'):
        clf = MLPClassifier()
        clf.fit(X_train, y_train)

    if(model == 'LinearSVC'):
        clf = LinearSVC()
        clf.fit(X_train, y_train)
        
    if(model == 'Perceptron'):
        clf = Perceptron()
        clf.fit(X_train, y_train)
        
    if(model == 'SVM'):
        clf = SGDClassifier()
        clf.fit(X_train, y_train)
        
    return clf

# Execute GS for the model we chose with the appropriate parameters
def grid_search_model(model, clf, X_train, y_train):
    
    if(model == 'KNN'):
        parameters = {'metric':['euclidean', 'manhattan', 'minkowski'],
                      'n_neighbors':[11, 20, 25, 30],
                      'weights':['uniform', 'distance']}
        
    if(model == 'DT'):
        parameters = {'criterion':['gini', 'entropy'],
                      'max_depth':[ 4, 5, 6, 7, 8],
                      'min_samples_leaf':[ 5, 6, 7, 8, 9], 
                      'min_samples_split':[10, 15, 20, 25]}
        
    if(model == 'MLP'):
            parameters = {'alpha':[0.0001, 0.0013, 0.0133], 
                          'solver':['lbfgs', 'sgd', 'adam']}
    
    if(model == 'LinearSVC'):
            parameters = {'penalty':['l2', 'l1'],
                          'loss':['hinge', 'squared_hinge']}
    
    if(model == 'Perceptron'):
            parameters = {'penalty':['l2', 'l1'],
                          'alpha':[0.0001, 0.0002, 0.0003]}
        
    if(model == 'SVM'):
        parameters = {'loss':['hinge', 'log', 'squared_hinge', 'perceptron'], 
                      'penalty':['l2', 'l1'], 
                      'alpha':[0.0001, 0.002, 0.0003]}
    
    GS_model = GridSearchCV(estimator=clf, param_grid=parameters, cv=5)
    GS_model.fit(X_train, y_train)
    return GS_model

# Create a list of models to chack
model_names = [
    'KNN',
    'DT',
    'MLP',
    'LinearSVC',
    'Perceptron',
    'SVM'
]

# Create variables to chack which model is the best for text prediction
f1_acc_li = []
bestScore = 0
bestModel = None

# --------------------------------------------------------------------------------- #

for model in range(len(model_names)):
    clf = train_model(model_names[model], X_train, y_train)
    GS_model = grid_search_model(model_names[model], clf, X_train, y_train)
    st.subheader(f'Model: {model_names[model]}\n')
    st.write(f'Best params: {GS_model.best_params_}\n')
    
    model_cv_score = cross_val_score(GS_model.best_estimator_, X_train, y_train, scoring=metrics.make_scorer(f1_score, average='macro'), cv=5)
    st.write(f'{model_names[model]} Cross Validation Scores: {model_cv_score}\n')
    st.write(f'{model_names[model]} Mean Accuracy: {model_cv_score.mean():.4f}\n')
    
    if(model_cv_score.mean() > bestScore):
        bestScore = model_cv_score.mean()
        bestModel = model_names[model]
    
    # Prediction
    y_pred = clf.predict(X_test)

    df_test['predicted_category'] = y_pred
    df_test['predicted_category'] = df_test['predicted_category'].replace({0:'f', 1:'m'})
    
    if (bestModel == model_names[model]):
        df_predicted = df_test[['test_example_id', 'predicted_category']]
    
    st.write("First 5 predictions:\n", df_test.head())
    st.write()
    st.write("Last 5 predictions:\n", df_test.tail())
    st.write()
    # Calculate the number of Males (assuming 1 represents Males in y_pred)
    male_count = np.count_nonzero(y_pred)
    st.markdown(f'<span style="color: green; font-weight: bold;">Males:</span> {male_count}', unsafe_allow_html=True)
    # Calculate the number of Females (assuming 0 represents Females in y_pred)
    female_count = y_pred.shape[0] - np.count_nonzero(y_pred)
    st.markdown(f'<span style="color: magenta; font-weight: bold;">Females:</span> {female_count}', unsafe_allow_html=True)
    st.write()
    yPredValues = format_yPredValues(y_pred)
    st.markdown(f"**y_pred values**:\n {yPredValues}", unsafe_allow_html=True)
    st.write('---------------------------------------------------------------')

st.write("The best model is:", bestModel)
st.write("The model f1 score is", bestScore)

# This is a DF that presents the gender predictions by the best model
st.write("Prediction of the best model")
st.write(df_predicted.head(10))
