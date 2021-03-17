import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def task1(df):
    sentiments_column = df['Sentiment']
    possible_sentiments = set(sentiments_column)
    print('all possible sentiment values: ' + str(possible_sentiments))
    second_most_popular_sent = sentiments_column.value_counts().sort_values(ascending=False).index[1]
    print('second most popular sentiment: ' + str(second_most_popular_sent))
    df_extremly_positive = df[df['Sentiment'] == 'Extremely Positive']
    date_with_most_extremely_positive_sent = df_extremly_positive.groupby(['TweetAt']).count().sort_values(
        ascending=False, by=['Sentiment']).index[0]
    print('date with most extremely positive sentiment: ' + str(date_with_most_extremely_positive_sent))
    # lower case
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: x.lower())
    # replace non alhabetical characters
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub('[^0-9a-zA-Z]+', ' ', x))
    # ensure that the words of a message are separated by a single whitespace
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: re.sub(' {2,}', ' ', x))

    return df


def count_occurences(dict, words):
    for word in words:
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            dict[word] = 0


def task2(df):
    # add TokenizedTweets column
    df.insert(5, 'TokenizedTweets', df['OriginalTweet'].apply(lambda x: x.split()))

    total_words_count = df['TokenizedTweets'].apply(lambda x: len(x)).sum()
    print('total number of words: ' + str(total_words_count))

    # number of all distinct words
    unique_words = set()
    df['TokenizedTweets'].apply(lambda x: [unique_words.add(word) for word in x])
    print('number of unique words: ' + str(len(unique_words)))

    # the 10 most frequent words in the corpus
    n_most_frequent_words(df, 10)

    # remove stop words, words with ≤ 2 characters
    df['TokenizedTweets'] = df['TokenizedTweets'].apply(
        lambda x: [word for word in x if len(word) > 2])

    total_words_count = df['TokenizedTweets'].apply(lambda x: len(x)).sum()
    print('total number of words after removing stop words: ' + str(total_words_count))

    # the 10 most frequent words in the corpus
    dict_most_frequent_words = n_most_frequent_words(df, 10)

    return df, dict_most_frequent_words


def n_most_frequent_words(df, n):
    dict = {}
    df['TokenizedTweets'].apply(lambda x: count_occurences(dict, x))
    most_frequent_words = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print('the 10 most frequent words: ' + str(most_frequent_words[0:n]))

    return dict


def task3(dict):
    plt.plot(sorted(dict.values(), key=lambda x: x, reverse=False))
    # plt.show()


def task4(df):
    corpus = np.array(df['OriginalTweet'])
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    y = np.array(df['Sentiment'])
    clf = MultinomialNB()
    clf.fit(X, y)

    print(clf.score(X, y))


if __name__ == '__main__':
    # start_time = time.time()

    df = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')

    df_converted = task1(df)

    df_tokenized, dict_most_frequent_words = task2(df_converted)

    # task 3
    task3(dict_most_frequent_words)

    # task 4
    task4(df_tokenized)

    # print("--- %s seconds ---" % (time.time() - start_time))
