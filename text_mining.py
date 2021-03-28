import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Returns the dataframe after applying operations specified in question 1.

# Parameters:
#    df (Dataframe):The dataframe which is to be applied operations

# Returns:
#    The dataframe after applying operations
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


# Counts number of occurences of words.

# Parameters:
#    dict (dict): The dict which is used to store information about occurences of a given words. Each word is added
#    to this dict as a key, where value corresponds to number of occurences
#    words (Iterable): collection of words to count
def count_occurrences(dict, words):
    for word in words:
        if word in dict:
            dict[word] = dict[word] + 1
        else:
            dict[word] = 0


# Returns the dataframe after applying operations specified in question 2 and dict with most frequent words.

# Parameters:
#    df (Dataframe):The dataframe which is to be applied operations

# Returns:
#    Dataframe after applying operations and dict with most fequent words 
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


# Returns the dictionary of words frequency

# Parameters:
#    df (Dataframe): The dataframe with column TokenizedTweets for which counting is applied
#    n (int): Number of most frequent words to be printed. For example n=10 will result in printing results for 10 most 
#    frequent words. This is however, independent of dict that is returned, which is not limited by this parameter.

# Returns:
#    dictionary of words frequency  
def n_most_frequent_words(df, n):
    dict = {}
    np.vectorize(count_occurrences)(dict, df['TokenizedTweets'])

    most_frequent_words = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print('the 10 most frequent words: ' + str(most_frequent_words[0:n]))

    return dict


# Create line chart of words frequency and saves it to file

# Parameters:
#    df_size (int): size of the dataframe. Equivalent of number of documents in dataset for which we plot the chart
#    words_frequencies_dict (dict):

# Returns:
#    void
def task3(words_frequencies_dict, df_size):
    # create new dict to obtain fraction of documents in a which a word appears
    words_fraction_appearing = {k: v / df_size for (k, v) in words_frequencies_dict.items()}

    # plot chart after sorting
    plt.plot(sorted(words_fraction_appearing.values(), key=lambda x: x, reverse=False))
    # save image to file
    plt.savefig('outputs/most_frequent.jpg')


# Create MultinomialNB that is to trained on dataframe and print its error rate.

# Parameters:
#    df (Dataframe): The dataframe used to traing the model. Need to have column 'OriginalTweet' that is used as
#    sample data and 'Sentiment' column which contains target value

# Returns:
#    void
def task4(df):
    # create count vectorizer
    cv = CountVectorizer()

    # transform data using the count vectorizer
    X = cv.fit_transform(np.array(df['OriginalTweet']))
    # transform target data to numpy array
    y = np.array(df['Sentiment'])

    # create MultinomialNB
    clf = MultinomialNB()
    # train the model
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
    task3(dict_most_frequent_words, len(df_tokenized))

    # # task 4
    task4(df_tokenized)

    print("--- %s seconds ---" % (time.time() - time_total))
