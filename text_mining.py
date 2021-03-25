import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def task1(df):
    # get all sentiments
    sentiments_column = df['Sentiment']
    # get unique sentiment values
    possible_sentiments = set(sentiments_column)
    print('all possible sentiment values: ' + str(possible_sentiments))

    # get values and number of occurences of a given value, then sort it and get second most occurring one
    second_most_popular_sent = sentiments_column.value_counts().sort_values(ascending=False).index[1]
    print('second most popular sentiment: ' + str(second_most_popular_sent))

    # filter df to retrieve samples with Extremely Positive sentiment only
    df_extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
    # group by date, sort it and retrieve the most popular one
    date_with_most_extremely_positive_sent = df_extremely_positive.groupby(['TweetAt']).count().sort_values(
        ascending=False, by=['Sentiment']).index[0]
    print('date with most extremely positive sentiment: ' + str(date_with_most_extremely_positive_sent))

    # convert to lower case
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()

    # replace non alphabetical characters
    df['OriginalTweet'] = df['OriginalTweet'].replace('[^0-9a-zA-Z]+', ' ', regex=True)

    # ensure that the words of a message are separated by a single whitespace
    df['OriginalTweet'] = df['OriginalTweet'].replace(' {2,}', ' ', regex=True)

    return df


def count_occurrences(dict, words):
    for word in words:
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            dict[word] = 0


def task2(df):
    # add TokenizedTweets column and insert split words
    df.insert(5, 'TokenizedTweets', df['OriginalTweet'].str.split())

    # count total number of words with repetitions
    total_words_count = df['TokenizedTweets'].str.len().sum()
    print('total number of words: ' + str(total_words_count))

    # number of all distinct words
    unique_words = set()
    df['TokenizedTweets'].apply(lambda x: [unique_words.add(word) for word in x])
    print('number of unique words: ' + str(len(unique_words)))

    # the 10 most frequent words in the corpus
    n_most_frequent_words(df, 10)

    stop_words = set(open('./data/text_data/stopwords.txt').read().splitlines())

    # remove stopwords or words with ≤ 2 characters
    df['TokenizedTweets'] = df['TokenizedTweets'].apply(
        lambda x: [word for word in x if len(word) > 2 and word not in stop_words])

    total_words_count = df['TokenizedTweets'].str.len().sum()
    print('total number of words after removing stop words: ' + str(total_words_count))

    # the 10 most frequent words in the corpus
    dict_most_frequent_words = n_most_frequent_words(df, 10)

    return df, dict_most_frequent_words


def n_most_frequent_words(df, n):
    dict = {}
    np.vectorize(count_occurrences)(dict, df['TokenizedTweets'])

    most_frequent_words = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print('the 10 most frequent words: ' + str(most_frequent_words[0:n]))

    return dict


def task3(words_frequencies_dict, df):
    df_size = len(df)
    mapped_dict = {k: v / df_size for (k, v) in words_frequencies_dict.items()}

    plt.plot(sorted(mapped_dict.values(), key=lambda x: x, reverse=False))
    plt.savefig('outputs/most_frequent.jpg')


def task4(df):
    cv = CountVectorizer()

    X = cv.fit_transform(np.array(df['OriginalTweet']))
    y = np.array(df['Sentiment'])

    clf = MultinomialNB()
    clf.fit(X, y)

    print('error rate: ' + str(1 - clf.score(X, y)))


if __name__ == '__main__':
    time_total = time.time()

    df = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')

    # task 1
    df_converted = task1(df)

    # task 2
    df_tokenized, dict_most_frequent_words = task2(df_converted)

    # # task 3
    task3(dict_most_frequent_words, df_tokenized)

    # # task 4
    task4(df_tokenized)

    print("--- %s seconds ---" % (time.time() - time_total))
