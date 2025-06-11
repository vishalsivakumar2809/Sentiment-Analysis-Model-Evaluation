import os
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from utils import process_tweet, get_accuracy

vader_folder = 'graphs/LogisticClassifier'
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

# Iteration Rate values based on sum of 5000
iterations = [50, 100, 500, 1000, 2500, 5000, 10000, 15000, 20000]

# Using validation set to select the best iteration rate
best_acc = 0
best_iter = 0
best_preds = None
accuracies = []
for iter in iterations:
    clf = LogisticRegression(max_iter=iter)
    clf.fit(X_train, train_y)
    val_preds = clf.predict(X_val)

    acc = get_accuracy(val_y, val_preds)

    if acc > best_acc:
        best_iter = iter
        best_acc = acc
        best_preds = val_preds
    accuracies.append(acc)

clf = LogisticRegression(max_iter=best_iter)
clf.fit(X_train, train_y)
test_preds = clf.predict(X_test)
test_acc = get_accuracy(test_y, test_preds)

print("-" * 100)
print("LOGISTIC REGRESSION CLASSIFIER RESULTS")
print("-" * 100)
print("Best Max Iteration: ", best_iter)
print("Accuracy Logistic Regression (Validation Data): {:.2%}".format(best_acc))
print("Accuracy Logistic Regression (Test Data): {:.2%}".format(test_acc))
print("-" * 100 + '\n')

plt.figure()
ConfusionMatrixDisplay.from_predictions(val_y, best_preds)
plt.title("Confusion Matrix (Validation Set)")
plt.savefig(f'{vader_folder}/confusionMatrixLogisticVal.jpg', dpi=300)
plt.close()

plt.figure()
ConfusionMatrixDisplay.from_predictions(test_y, test_preds)
plt.title("Confusion Matrix (Test Set)")
plt.savefig(f'{vader_folder}/confusionMatrixLogisticTest.jpg', dpi=300)
plt.close()

plt.figure()
plt.plot(iterations, accuracies, marker='o')
plt.xlabel('Smoothing value (max_iter)')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Changing max_iter on Accuracy')
plt.savefig(f'{vader_folder}/val_accuracy_vs_alpha.jpg', dpi=300)
plt.close()



