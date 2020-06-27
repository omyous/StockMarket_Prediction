
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from machine_learning.sentiments_analysis import *
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999

class Custom_dataset():
    def __init__(self):
        self.tweets_path = "data/google_tweets_.csv"
        self.prices_path = "data/google.csv"
        self.tweets = None
        self.prices = None

    def get_data(self):
        self.tweets = pd.read_csv(self.tweets_path)
        self.prices = pd.read_csv(self.prices_path)


    def clean_data(self):
        self.get_data()
        self.clean_tweets()
        self.clean_prices()



    def get_vectors(self,strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    def get_cosine_sim(self,strs):
        vectors = [t for t in self.get_vectors(strs)]
        return cosine_similarity(vectors)


    def tweets_sim(self,df):
        df_ = df
        print(df_)
        # create a clean dataframe
        clean_df2 = pd.DataFrame(columns=["date", "text"])
        indexes = df_.index.values.tolist()
        i = 0
        while i < len(indexes):
            if df_.loc[indexes[i]]["text"] != "str":
                j = i + 1
                while j < len(indexes):
                    if df_.loc[indexes[j]]["text"] != "str":
                        df_.loc[indexes[i]]["text"] = " ".join(sent_tokenize(df_.loc[indexes[i]]["text"]))
                        df_.loc[indexes[j]]["text"] = " ".join(sent_tokenize(df_.loc[indexes[j]]["text"]))

                        result = self.get_cosine_sim(df_.iloc[[i, j]]["text"])

                        if result[0, 1] >= .5:
                            print(result, "\n")
                            print("SIMILAIRE: ", df_.iloc[[i, j]]["text"])
                        else:
                            print("DIFFERENT: ", df_.iloc[[i, j]]["text"])
                            print(result)
                            clean_df2 = clean_df2.append({'date': df.loc[indexes[j]]["date"],
                                                        'text': df.loc[indexes[j]]["text"]},
                                                       ignore_index=True)

                    df_.loc[indexes[j]]["text"] = "str"
                    j += 1
                clean_df2 = clean_df2.append({'date': df.loc[indexes[i]]["date"],
                                            'text': df.loc[indexes[i]]["text"]},
                                           ignore_index=True)
                df_.loc[indexes[i]]["text"] = "str"
            i += 1

        return clean_df2

    def drop_tweetsduplicates(self):
        dates = self.tweets["date"].unique()
        clean_df1 = pd.DataFrame(columns=["date", "text"])
        for d in dates:
            df_ = self.tweets[self.tweets["date"] == d]
            clean_df1 = clean_df1.append(self.tweets_sim(df_))
        return clean_df1

    def clean_tweets(self):

        self.tweets["date"] = pd.to_datetime(self.tweets["date"], format='%Y/%m/%d').dt.date

        self.tweets = self.drop_tweetsduplicates()
        #drop any duplicate post
        self.tweets = self.tweets.drop_duplicates("text")
        self.tweets['expand'] = self.tweets.apply(lambda x: '. '.join([x['text']]), axis=1)
        self.tweets = self.tweets.groupby('date')['expand'].apply(list)
        #self.tweets["date"] = self.tweets.index.values
        self.tweets = pd.DataFrame(data=self.tweets.values, index=self.tweets.index, columns=["text"])
        #create one daily tweet  intead of many ones
        text = [' '.join(sentence) for sentence in self.tweets["text"]]
        self.tweets["text"] = text

        self.tweets.to_csv("data/clean_tweets.csv", index=True)

    def clean_prices(self):

        #delete zeros columns
        self.prices = self.prices.loc[:, (self.prices != 0).any(axis=0)]
        self.prices["Date"] = pd.to_datetime(self.prices["Date"], format='%Y/%m/%d').dt.date

        # drop duplicate
        self.prices = self.prices.drop_duplicates(subset='Date', keep="last")
        self.prices.to_csv("data/clean_prices.csv", index=False)

    def merge_data(self):
        tweets = pd.read_csv("data/tweets_scores.csv")
        #tweets["date"]=pd.to_datetime(tweets["date"], format='%Y/%m/%d').dt.date
        prices = pd.read_csv("data/clean_prices.csv")

        tweets_dates = list(tweets["date"])
        prices_dates = list(prices["Date"])

        for i in prices_dates:
            if not i in tweets_dates:
                print("insérer ",i," dans tweets")
                tweets = tweets.append({'date': i,
                                        'positive': tweets["positive"].median(),
                                        'negative':tweets['negative'].median()
                                        },
                                       ignore_index=True)

        print("####################################\n\n")

        for i in tweets_dates:
            if not i in prices_dates:
                print("insérer ",i, " dans prices")
                prices = prices.append({'Date': i,
                                        'High': prices["High"].median(),
                                        'Low':prices['Low'].median(),
                                        'Open':prices['Open'].median(),
                                        'Close':prices['Close'].median(),
                                        'Volume':prices['Volume'].median(),
                                        'Adj Close':prices['Adj Close'].median()
                                        },
                                       ignore_index=True
                                       )


        tweets=tweets.sort_values("date", ascending=True)
        tweets.to_csv("data/tweets_scores_.csv", index=False)
        prices=prices.sort_values("Date", ascending=True)
        prices.to_csv("data/clean_prices_.csv", index=False)
        df = tweets.rename(columns={'date': 'Date'})
        df = prices.merge(df, on="Date")
        df.to_csv("data/merged.csv", index=False)



if __name__ == "__main__":
    df = Custom_dataset()
    df.clean_data()


