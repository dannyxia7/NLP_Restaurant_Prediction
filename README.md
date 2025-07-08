# NLP_Restaurant_Prediction
I am attempting to predict the ethnic type of restaurants based on a variety of columns in the dataset. The most important column includes yelp reviews of the various restaurants, which is where NLP comes into play. 

# Description

In this section, you can find descriptions of each of the resources you will be using for your project.
train.csv

It contains all the training data that you can use in this challenge. The first column id provides you with the unique key to identify the restaurants. Different columns show different features/attributes of a restaurant, including free texts, numerical features, and categorical features. There are also many missing values. So please conduct some exploratory data analysis (EDAs) first before you work on feature engineering. 
test.csv

It contains all the restaurants that you need to predict their types. The format is the same as the train.csv, except that the label column has been removed. 

baseline.ipynb

It contains a logistic regression model trained on document vectors computed by averaging word vectors of its constituent words from reviews. By running this notebook, you will be able to get a predicted.csv as the output. The baseline gives an example of using restaurant reviews only, but you are free to utilize any attributes/features in your approach. You should aim for better performance. 
