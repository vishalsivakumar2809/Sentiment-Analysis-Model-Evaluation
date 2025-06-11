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

# Report
| Model         | Validation Performance (Accuracy) | Test Performance (Accuracy) | 
|---------------|----------|-----------|
| Linear Regression | 66.25%   | 64.40%    | 
| Logistic Regression | 71.25%    | 72.50%    | 
| Naive Bayes   | 99.50%     | 99.40%     | 
|---------------|----------|-----------|

| Model         | Positive Accuracy  | Negative Accuracy  | Overall Accuracy |
|---------------|----------|-----------|-----------|
| VADER         | -    | 84.24%     | 
|---------------|----------|-----------|-----------|

# Conclusion

VADER demonstrated strong performance on social media content by leveraging its lexicon of slang, 
emojis, and acronyms. However, it showed reduced accuracy when detecting negative sentiment, likely due to the subtlety of some negative expressions.
Among the machine learning models, Naive Bayes achieved the highest performance.
Its success is likely due to the dataset’s clear sentiment cues and balanced class distribution,
which aligns well with the independence assumption of the model. Logistic Regression
outperformed Linear Regression, as expected, due to its probabilistic nature. However, both
regression-based models were affected by the limited sample size and the large number of
features, which reduced their ability to generalize.
Overall, the results highlight that rule-based models can offer fast and
domain-relevant performance with minimal training, while machine learning models are
capable of higher accuracy under the right data conditions. For real-world deployment,
especially on noisy or imbalanced datasets, further validation and more robust modeling
techniques may be necessary.

# References
