import pandas as pd
import re


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


if __name__ == '__main__':
    df = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')

    df_converted = task1(df)

    # add TokenizedTweets column
    df_converted.insert(5, 'TokenizedTweets', df_converted['OriginalTweet'].apply(lambda x: x.split()))

    total_words_count = df_converted['TokenizedTweets'].apply(lambda x: len(x)).sum()
    print('total number of words: ' + str(total_words_count))
