import os
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import twitter_samples
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import get_accuracy

vader_folder = 'graphs/Vader'
if not os.path.exists(vader_folder):
    os.makedirs(vader_folder)

# Download twitter samples if not already present
nltk.download('twitter_samples')

# Load the positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Initialize the VADER analyzer
analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    compound = score["compound"]
    if compound > 0:
        return 1
    else:
        return 0

# positive accuracy calculation
positive_preds = np.array([vader_sentiment(tweet) for tweet in all_positive_tweets])
positive_true = np.ones(len(all_positive_tweets), dtype=int)
accuracy_positive = get_accuracy(positive_preds, positive_true)

# negative accuracy calculation
negative_preds = np.array([vader_sentiment(tweet) for tweet in all_negative_tweets])
negative_true = np.zeros(len(all_negative_tweets), dtype=int)
accuracy_negative = get_accuracy(negative_preds, negative_true)

# overall accuracy calculation
all_preds = np.concatenate([positive_preds, negative_preds])
all_true = np.concatenate([positive_true, negative_true])
overall_accuracy = get_accuracy(all_preds, all_true)

# Print results
print("-" * 100)
print("VADER RESULTS")
print("-" * 100)
print("Positive Accuracy: {:.2%}".format(accuracy_positive))
print("Negative Accuracy: {:.2%}".format(accuracy_negative))
print("Overall Accuracy: {:.2%}".format(overall_accuracy))
print("-" * 100 + '\n')

# Plot the accuracies
labels = ['Positive', 'Negative', 'Overall']
accuracies = [accuracy_positive, accuracy_negative, overall_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=['red', 'yellow', 'blue'])

plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('VADER Sentiment Accuracy')

# Annotate each bar with the corresponding accuracy percentage
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2%}', ha='center', va='bottom')

plt.savefig(f'{vader_folder}/Sentiment_Analysis_Accuracy_Vader.jpg', dpi=300)
