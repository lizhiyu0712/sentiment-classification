Your objective is to perform sentiment analysis on the reviews. To achieve this, you will train and test a Bag-of-Words (BoW) text classifier using the review texts and their corresponding rating values. The rating values will be categorized into three sentiment categories: negative (ratings 1 & 2), neutral (rating 3), and positive (ratings 4 & 5). These sentiment categories will be encoded as follows: negative as 0, neutral as 1, and positive as 2. The resulting column containing the binned ratings will be named "Sentiment," as described in Table 1.

However, the ratings exhibit a significant imbalance, with a considerably higher number of positive (2) ratings compared to negative (0) ratings. To address this issue, you will need to exclude the positive ratings from the dataset to achieve a balanced distribution, where the number of negative, neutral, and positive ratings is approximately equal.
