---
title: "Using Naive Bayes as a Baseline Model"
summary: This is a short demo on how to implement a NB model in SKlearn
date: 2020-07-20T00:50:00
categories: ["Quick Solves"]
tags: ["Naive Bayes","sklearn", "Machine Learning Learning"]
authors:
- admin
draft: false
image:
  focal_point: "Center"
  
---

# Naive bayes

The other day I had to prepare a class showing the benefits of using Naive Bayes. I have to say this is not a super powerful model, mainly because it makes assumptions that are most of the time not true. Nevertheless, I noticed this can be an excellent way to create a baseline model. It is easy, not very complicated to implement and the best thing is that is super fast.

As you may know, this model is based on the Bayes Theorem, namely:

$$P[B/A] = \frac{P[A / B] \cdot P[B]}{P[A]}$$

I´m not gonna demonstrate why this happens (basically because I don´t know how), but this models is based on probabilities that are calculated out of the dataset itself. In order to assign a class the class is calculated as:

$$y = k = argmax\, P[y = k] \cdot \prod{}_{i = 1}^p P[X_i/y = k]$$

Where the class is denoted by the maximum probability (between the different classes) of the product of the a priori probability of y and the different likelihood of Variables $X_i$ given y = k. 

This example will be implemented using a lyrics dataset that I found [here](https://github.com/hiteshyalamanchili/SongGenreClassification/blob/master/dataset/english_cleaned_lyrics.zip). Shoutouts to Hitesh Yalamanchili for making this data available.

So this naive Bayes model will be implemented in Python trying to Predict what genre a song belongs to by using its lyrics. So the implementation in Python looks like this:

# Importing Data

When trying to import the data I noticed this has the following form:

![](data_cap.PNG)

For some reason there is a duplicated Index. In order to avoid a weird `Unnamed: 0` column I had to use `names` argument in `pd.read_csv()` to declare the actual column names to import. Even by doing that the DataFrame was imported as a double Index dataset so I had to remove one of the index using `.reset_index()`.

> Note: In order to make the dataset manageable for demonstration purposes only I decided to use only four genres: Rock, Pop, Hip-Hop and Metal.


```python
%%time
import pandas as pd
df = pd.read_csv('english_cleaned_lyrics.csv', header = 0, names = ['song','year','artist','genre', 'lyrics'], index_col = None).reset_index(level = 1, drop = True)
df.query('genre in ["Rock","Pop","Hip-Hop","Metal"]', inplace = True)
df
```

    Wall time: 3.32 s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song</th>
      <th>year</th>
      <th>artist</th>
      <th>genre</th>
      <th>lyrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ego-remix</td>
      <td>2009</td>
      <td>beyonce-knowles</td>
      <td>Pop</td>
      <td>Oh baby how you doing You know I'm gonna cut r...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>then-tell-me</td>
      <td>2009</td>
      <td>beyonce-knowles</td>
      <td>Pop</td>
      <td>playin everything so easy it's like you seem s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>honesty</td>
      <td>2009</td>
      <td>beyonce-knowles</td>
      <td>Pop</td>
      <td>If you search For tenderness It isn't hard to ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>you-are-my-rock</td>
      <td>2009</td>
      <td>beyonce-knowles</td>
      <td>Pop</td>
      <td>Oh oh oh I oh oh oh I If I wrote a book about ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>black-culture</td>
      <td>2009</td>
      <td>beyonce-knowles</td>
      <td>Pop</td>
      <td>Party the people the people the party it's pop...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>362210</th>
      <td>photographs-you-are-taking-now</td>
      <td>2014</td>
      <td>damon-albarn</td>
      <td>Pop</td>
      <td>When the photographs you're taking now Are tak...</td>
    </tr>
    <tr>
      <th>362211</th>
      <td>you-and-me</td>
      <td>2014</td>
      <td>damon-albarn</td>
      <td>Pop</td>
      <td>I met Moko jumbie He walks on stilts through a...</td>
    </tr>
    <tr>
      <th>362212</th>
      <td>hollow-ponds</td>
      <td>2014</td>
      <td>damon-albarn</td>
      <td>Pop</td>
      <td>Chill on the hollow ponds Set sail by a kid In...</td>
    </tr>
    <tr>
      <th>362213</th>
      <td>the-selfish-giant</td>
      <td>2014</td>
      <td>damon-albarn</td>
      <td>Pop</td>
      <td>Celebrate the passing drugs Put them on the ba...</td>
    </tr>
    <tr>
      <th>362214</th>
      <td>hostiles</td>
      <td>2014</td>
      <td>damon-albarn</td>
      <td>Pop</td>
      <td>When the serve is done And the parish shuffled...</td>
    </tr>
  </tbody>
</table>
<p>178054 rows × 5 columns</p>
</div>



## Feature Extraction

In this step, we´ll use the `CountVectorizer()` class to provide a Occurrence Matrix. In this Matrix every row will be a Document, in this case a song, whereas every column is a Word. If a word ocurrs in the Document the is denoted by a 1. The only processing to the data is stopwords removal, that is removing all the words that are too common that end up adding noise to the analysis.

We'll then use the word ocurrences as predictors for the genre. The predictor will look like this:


```python
%%time
from sklearn.feature_extraction.text import CountVectorizer

c_vec = CountVectorizer(stop_words = 'english', max_features = 20000) ## I´m removing english stopwords, and setting the max number of predictors to 20000 to avoid my computer to crush.
vectorizer = c_vec.fit_transform(df['lyrics']) 
# Transform output into pandas Df for visualization
pd.DataFrame(vectorizer.toarray(), columns = c_vec.get_feature_names()) 
```

    Wall time: 35.6 s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>02</th>
      <th>03</th>
      <th>05</th>
      <th>06</th>
      <th>07</th>
      <th>09</th>
      <th>10</th>
      <th>100</th>
      <th>...</th>
      <th>zones</th>
      <th>zonin</th>
      <th>zoo</th>
      <th>zoom</th>
      <th>zoomin</th>
      <th>zoovie</th>
      <th>zoovier</th>
      <th>zoowap</th>
      <th>zu</th>
      <th>zulu</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>178049</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>178050</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>178051</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>178052</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>178053</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>178054 rows × 20000 columns</p>
</div>



## Setting up the Model

The model is usper easy to set up. We just need to import `MultinomialNB` since this is a multiclass Prediction Model. Additionally, we´ll import `train_test_split()` to split the data into train and test, `Pipeline()` to create the Model Pipeline (the steps to come up with the model) and `classification_report()` to measure model performance.


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
```


```python
# X will be song lyrics and y is the genre. We´ll split the data using 40% for test purposes.
X_train, X_test, y_train, y_test = train_test_split(df['lyrics'], df['genre'], test_size = 0.4, random_state = 123) 
```

Then a pipeline will be set using 2 steps. The first one being the `CountVectorizer()` named 'cv' and the `MultinomialNB()` model named 'nb':


```python
%%time
text_clf = Pipeline(steps = [
    ('cv', CountVectorizer(stop_words = 'english', max_features = 20000)),
    ('nb', MultinomialNB(alpha = 0.1))
])
text_clf.fit(X_train, y_train) # fitting the Pipeline
#predicting using the Test Set to measure performance
y_pred = text_clf.predict(X_test)
```

    Wall time: 26.4 s
    

The first thing to notice is that even having 178K rows and 20000 predictors the model fits in under 30 seconds. FAST!

Now when it comes to results, it is not a terrible model, it has a 63% of accuracy and the Macro F1 is 62%. Not bad for just using a couple of lines of code.


```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
         Hip-Hop       0.72      0.77      0.74      9062
           Metal       0.48      0.75      0.59      8551
             Pop       0.42      0.53      0.47     13582
            Rock       0.78      0.60      0.68     40027
    
        accuracy                           0.63     71222
       macro avg       0.60      0.66      0.62     71222
    weighted avg       0.66      0.63      0.63     71222
    
    

## Improving the Model

In order to improve the model, we could run a GridSearch trying to play around with the alpha smoothing parameter that NB has. By adjusting this parameter correctly we could easily improve a bit the model without too much effort.

In this case we´ll run a Grid using values from 0 to 1, as shown below. Additionally we´ll use the 'f1_macro' as the metric to choose the best model using a 5-Fold Cross Validation.


```python
%%time
from sklearn import set_config
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
#Grilla de parámetros a buscar
parameters = {'nb__alpha': [0, 0.001, 0.01, 0.1, 0.5, 1] }

text_clf = Pipeline(steps = [
    ('cv', CountVectorizer(stop_words = 'english')),
    ('nb', MultinomialNB())
])


searchCV = GridSearchCV(text_clf, parameters, n_jobs = -1, scoring = 'f1_macro', cv = 5) # 5 Fold CV optimizando el modelo por f1 macro
searchCV.fit(X_train, y_train)

```

    Wall time: 3min 27s
    
    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('cv',
                                            CountVectorizer(stop_words='english')),
                                           ('nb', MultinomialNB())]),
                 n_jobs=-1, param_grid={'nb__alpha': [0, 0.001, 0.01, 0.1, 0.5, 1]},
                 scoring='f1_macro')



The GridSearch takes around 3 minutes to run 6 models using 5-Fold CV. And we can inmediately notice, small improvements for the best model:


```python
best_nb = searchCV.best_estimator_  # Extracting Best Model
y_pred = best_nb.predict(X_test) # Predicting the Test Set
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
         Hip-Hop       0.73      0.77      0.75      9062
           Metal       0.56      0.70      0.62      8551
             Pop       0.45      0.49      0.47     13582
            Rock       0.76      0.69      0.73     40027
    
        accuracy                           0.66     71222
       macro avg       0.63      0.66      0.64     71222
    weighted avg       0.67      0.66      0.67     71222
    
    

We can inmediately see that:

* Overall Accuracy improves 3%.
* F1 macro average improved 2%. 
* The Rock category is the one that improves the most from 68 to 73%.
* There is a trade off, even though some classess improve we can see that Metal decreases, whereas Pop keep the same results.

Finally we can check that the best is achieved when using alpha equals to 1.


```python
best_nb.named_steps.nb.get_params()
```




    {'alpha': 1, 'class_prior': None, 'fit_prior': True}



This is just a short example on how to set up a baseline model. Hopefully this can be useful for you.

See you next time!!
