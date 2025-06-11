import os
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from utils import process_tweet, get_accuracy

vader_folder = 'graphs/LinearClassifier'
if not os.path.exists(vader_folder):
    os.makedirs(vader_folder)

nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:3000]
val_pos = all_positive_tweets[3000:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:3000]
val_neg = all_negative_tweets[3000:4000]

train_x = train_pos + train_neg
val_x = val_pos + val_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))
val_y = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))

train_x_clean = [' '.join(process_tweet(tweet)) for tweet in train_x]
val_x_clean = [' '.join(process_tweet(tweet)) for tweet in val_x]
test_x_clean = [' '.join(process_tweet(tweet)) for tweet in test_x]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
X_train = vectorizer.fit_transform(train_x_clean)  # Learn vocabulary & transform training data
X_val = vectorizer.transform(val_x_clean)          # Just transform validation data
X_test = vectorizer.transform(test_x_clean)        # Just transform test data

linreg = LinearRegression()
linreg.fit(X_train, train_y)
val_preds = linreg.predict(X_val) >= 0.5
test_preds = linreg.predict(X_test) >= 0.5

print("-" * 100)
print("LINEAR REGRESSION CLASSIFIER RESULTS")
print("-" * 100)
print("Accuracy Linear Regression (Validation Data): {:.2%}".format(get_accuracy(val_y, val_preds)))
print("Accuracy Linear Regression (Test Data): {:.2%}".format(get_accuracy(test_y, test_preds)))
print("-" * 100 + '\n')

plt.figure()
ConfusionMatrixDisplay.from_predictions(val_y, val_preds)
plt.title("Confusion Matrix (Validation Set)")
plt.savefig(f'{vader_folder}/confusionMatrixLinearVal.jpg', dpi=300)
plt.close()

plt.figure()
ConfusionMatrixDisplay.from_predictions(test_y, test_preds)
plt.title("Confusion Matrix (Test Set)")
plt.savefig(f'{vader_folder}/confusionMatrixLinearTest.jpg', dpi=300)
plt.close()
