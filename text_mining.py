import pandas as pd
import matplotlib.pyplot as plt
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

    df_extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
    date_with_most_extremely_positive_sent = df_extremely_positive.groupby(['TweetAt']).count().sort_values(
        ascending=False, by=['Sentiment']).index[0]
    print('date with most extremely positive sentiment: ' + str(date_with_most_extremely_positive_sent))

    # lower case
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
    # add TokenizedTweets column
    df.insert(5, 'TokenizedTweets', df['OriginalTweet'].str.split())

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

# check if ok "while the vertical axis indicates the fraction of documents in a which a word appears"
def task3(dict):
    plt.plot(sorted(dict.values(), key=lambda x: x, reverse=False))
    plt.savefig('outputs/most_frequent.jpg')
    # plt.show()


def task4(df):
    cv = CountVectorizer()

    X = cv.fit_transform(np.array(df['OriginalTweet']))
    y = np.array(df['Sentiment'])

    clf = MultinomialNB()
    clf.fit(X, y)

    print('error rate: ' + str(1 - clf.score(X, y)))


if __name__ == '__main__':
    time_total = time.time()
    start_time = time_total

    df = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')
    print("reading data --- %s seconds ---" % (time.time() - start_time))

    # task 1
    start_time = time.time()
    df_converted = task1(df)
    print("task1 --- %s seconds ---" % (time.time() - start_time))

    # task 2
    start_time = time.time()
    df_tokenized, dict_most_frequent_words = task2(df_converted)
    print("task2 --- %s seconds ---" % (time.time() - start_time))

    # # task 3
    start_time = time.time()
    task3(dict_most_frequent_words)
    print("task3 --- %s seconds ---" % (time.time() - start_time))

    # # task 4
    start_time = time.time()
    task4(df_tokenized)
    print("task4 --- %s seconds ---" % (time.time() - start_time))

    print("--- %s seconds ---" % (time.time() - time_total))
