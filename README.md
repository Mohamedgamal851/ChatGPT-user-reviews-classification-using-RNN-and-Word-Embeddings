# ChatGPT-user-reviews-classification-using-RNN-and-Word-Embeddings
This is a Python code that classify ChatGPT user reviews using RNN and Word Embeddings

## Dataset: 
ChatGPT user reviews dataset; This dataset contains user reviews for the
ChatGPT application, with a total of 2000+ reviews attached with the assignment
“chatgpt_reviews.csv”. The sentiment labels are distributed as follows: Positive Reviews:
1,028 (45%) and Negative Reviews: 1,264 (55%).
# Requirements [8 grades]:

## Data Pre-processing:
to clean your data and provide a valid dataset for the models
to be trained, like removing special characters, removing stopwords using NLTK,
drop null values.
## Data Splitting and labels mapping: 
apply data splitting for your dataset; 80% as
training set and 20% as testing set and map labels to integers.
## Word Embedding (using keras): 
build your vocabulary by extracting and indexing
unique words, convert each review to a sequence of indices, then apply sequence
padding to have all sequences of the same length in preparation for input to the
embedding layer.
## Model Training (using keras): 
You will train two models simpleRNN and LSTM and
print the accuracy for each model on testing data.
## Bonus [2 grades]:
Provide a report that shows model summary of each model and the best
hyperparameters for each model (splitting ratio, sequence padding length ... ) with
a table showing the accuracy against each parameter (i.e. 80% 20% ratio, 70% 30%
ratio, and same for sequence padding length).
