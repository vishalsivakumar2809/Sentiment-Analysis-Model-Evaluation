import os
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import twitter_samples
from utils import process_tweet, get_accuracy

vader_folder = 'graphs/NaiveBayes'
if not os.path.exists(vader_folder):
    os.makedirs(vader_folder)

nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

def find_all_frequencies(tweets, ys):
    frequency_d = {}

    for tweet, y in zip(tweets, ys):
        for word in process_tweet(tweet):
                pair = (word, y)
                if pair in frequency_d:
                    frequency_d[pair] += 1
                else:
                    frequency_d[pair] = frequency_d.get(pair, 1)

    return frequency_d

def train_naive_bayes(freqs, train_y, alpha=1):
    log_likelihood = {}
    log_prior = 0

    unique_words = set([pair[0] for pair in freqs.keys()])
    V = len(unique_words)

    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[(pair)]
        
        else:
            N_neg += freqs[(pair)]

    D = train_y.shape[0]
    D_pos = sum(train_y)
    D_neg = D - D_pos

    log_prior = np.log(D_pos) - np.log(D_neg)

    for word in unique_words:
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        p_w_pos = (freq_pos + alpha) / (N_pos + V)
        p_w_neg = (freq_neg + alpha) / (N_neg + V)

        log_likelihood[word] = np.log(p_w_pos / p_w_neg)
    
    return log_prior, log_likelihood

def naive_bayes_predict(tweet, logprior, log_likelihood):
    word_l = process_tweet(tweet)
    p = 0 
    p += logprior

    for word in word_l:
        if word in log_likelihood:
            p += log_likelihood[word]
    
    return p 

def predict_all(tweets, logprior, log_likelihood):
    return np.array([1 if naive_bayes_predict(tweet, logprior, log_likelihood) > 0 else 0 for tweet in tweets])

# Make all required data for training, validation, and test
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

freqs = find_all_frequencies(train_x, train_y)
logprior, loglikelihood = train_naive_bayes(freqs, train_y)

# Smoothing Rate values on powers of 2
alphas = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 512, 1024]

# Using validation set to select the best smoothing rate
best_acc = 0
best_alpha = 0
accuracies = []
for alpha in alphas:
    logprior, loglikelihood = train_naive_bayes(freqs, train_y, alpha)
    preds = predict_all(val_x, logprior, loglikelihood)
    acc = get_accuracy(preds, val_y)

    if acc > best_acc:
        best_alpha = alpha
        best_acc = acc
    accuracies.append(acc)

# Use the best smoothing rate for the test set
logprior, loglikelihood = train_naive_bayes(freqs, train_y, best_alpha)
preds_test = predict_all(test_x, logprior, loglikelihood)
acc_test = get_accuracy(preds_test, test_y)

print("-" * 100)
print("NAIVE BAYES RESULTS")
print("-" * 100)
print("Best Smoothing Rate: ", best_alpha)
print("Accuracy Naive Bayes (Validation Data): {:.2%}".format(best_acc))
print("Accuracy Naive Bayes (Test Data): {:.2%}".format(acc_test))
print("-" * 100 + '\n')

plt.plot(alphas, accuracies, marker='o')
plt.xlabel('Smoothing value (alpha)')
plt.ylabel('Validation Accuracy')
plt.title('Effect of Laplace Smoothing on Accuracy')
plt.savefig(f'{vader_folder}/val_accuracy_vs_alpha.jpg', dpi=300)

