# Sentiment-Analysis-Model-Evaluation

In this project, I implement a Twitter Sentiment Analysis system using a dataset of
approximately 10,000 pre-labeled tweets, consisting of both positive and negative samples
from the NLTK corpus. Preprocessing techniques such as stemming and lemmatization are 
applied uniformly to the data to ensure consistency across all models. Additionally, all 
reported accuracies are calculated by the utility functions contained in utils.py.

Two types of classifiers are explored:
1. Rule-Based Classifier: This method is evaluated using VADER (Valence Aware Dictionary
and sEntiment Reasoner), a lexicon and rule-based sentiment analysis tool designed for
social media text.

2. Machine Learning Classifiers: These learned models (Linear Classifier, Logistic Classifier,
and Naive Bayes) are trained on the bag-of-words representation of the tweet corpus, aiming to
generalize from training data to make more flexible predictions on unseen text. 

This project is run in a virtual environment (recommended for dependency
management, though not mandatory if all required packages are already installed).
All required dependencies are listed in requirements.txt and will be installed
automatically during environment setup.

# Commands
```
bash scripts/create_env.sh # creates virtual environment
source myenv/bin/activate # opens virtual environment
bash scripts/run_all_programs.sh # runs all models
```

# Demo

# Findings

# References
