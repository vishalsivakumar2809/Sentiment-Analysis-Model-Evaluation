import re
import nltk
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
nltk.download('twitter_samples')

stopwords_english = stopwords.words('english')
punctuations = string.punctuation
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

def remove_hyperlinks_marks_styles(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet

def tokenize_tweet(tweet):
    return tokenizer.tokenize(tweet)

def remove_stopwords_punctuations(tweet_tokens):
    return [word for word in tweet_tokens if word not in stopwords_english and word not in punctuations]

def get_stem(tweets_clean):
    return [stemmer.stem(word) for word in tweets_clean]

def process_tweet(tweet):
    tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens = tokenize_tweet(tweet)
    tweets_clean = remove_stopwords_punctuations(tweet_tokens)
    return get_stem(tweets_clean)

def get_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)