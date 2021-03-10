import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/text_data/Corona_NLP_train.csv', encoding='latin-1')

    # print data.head()
    sentiments_column = data['Sentiment']
    possible_sentiments = set(sentiments_column)
    print('all possible sentiment values: '+str(possible_sentiments))

    second_most_popular_sent = sentiments_column.value_counts().sort_values(ascending=False).index[1]
    print('second most popular sentiment: '+str(second_most_popular_sent))

    df_extremly_positive = data[data['Sentiment'] == 'Extremely Positive']
    date_with_most_extremely_positive_sent = df_extremly_positive.groupby(['TweetAt']).count().sort_values(
        ascending=False, by=['Sentiment']).index[0]
    print('date with most extremely positive sentiment: '+str(date_with_most_extremely_positive_sent))



