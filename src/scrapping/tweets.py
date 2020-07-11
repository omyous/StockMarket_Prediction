import GetOldTweets3 as got
from pathlib import Path
import datetime as dt
import pandas as pd

class Tweets():
    def __init__(self, company: str = "Google business tech"):
        self.company = company
        #check if the file that contains the last scrapping data exists
        self.start_path = Path("data/tweet_start_file.txt")

        if self.start_path.is_file():
            self.file = open(self.start_path, 'r+')
            self.start = self.file.read()
            if self.start != '':
                self.start = dt.date(int(self.start[:4]), int(self.start[5:7]), int(self.start[8:10]))
                self.file.close()
            else:
                self.save_default_date()
        else:
            self.save_default_date()

        #self.end contain the ending scrapping date
        self.end = dt.datetime.today()

        self.tweetCriteria = got.manager.TweetCriteria().setQuerySearch(self.company) \
        .setSince(str(self.start)) \
        .setUntil(str(self.end)) \
        .setTopTweets(False)\
        .setMaxTweets(10)\
        .setLang('en')

    def get_tweets(self):
        years: list = ["2020"]#2018,2019
        months: list = ["01","02","03","04"]#"01","02","03","04","05","06","07","08","09","10","11","12"
        days: list = ["01","02","03","04","05", "06", "07", "08", "09", "10",
                      "11", "12", "13", "14","15", "16", "17", "18","19",
                      "20", "21","22","23", "24", "25", "26", "27", "28", "29",
                      "30", "31"
                      ]

        df = pd.read_csv("data/google_tweets_.csv")
        for y in years:
            for m in months:
                for d in days:
                    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(self.company) \
                        .setSince(y+"-"+m+"-"+d) \
                        .setUntil(y+"-"+m+"-"+str(int(d)+1)) \
                        .setTopTweets(False) \
                        .setMaxTweets(150) \
                        .setLang('en')
                    tweets=got.manager.TweetManager.getTweets(tweetCriteria)
                    for tt in tweets:
                        df = df.append({'date': tt.date, 'text': tt.text}, ignore_index=True)
                        print(tt.text)

        df.to_csv("data/google_tweets_.csv", index=False)


    #after the scrapping, we have to update the start date to get the test data afterwhile
    def update_start(self):
        self.file = open(self.start_path, "w+")
        self.file.write(str(self.end))
        self.file.close()
    def save_default_date(self):
        self.start = dt.date(2019, 1, 1)
        self.file = open(self.start_path, 'w+', encoding="utf-8")
        self.file.write(str(self.start))
        self.file.close()


    def update_tweet_hist(self):
        #get the last scrapping date in order to scrapp the latest tweets sice that day until today
        start= open(self.start_path, "r+").read()
        start= dt.date(int(start[:4]), int(start[5:7]), int(start[8:10]))

        #load the historical tweets
        df = pd.read_csv("data/google_tweets_.csv")

        months: list = [str(start.month)]

        if dt.date.today().month != start.month:
            months.append(str(dt.date.today().month))
            days:list =["01","02","03","04","05", "06", "07", "08", "09", "10",
                      "11", "12", "13", "14","15", "16", "17", "18","19",
                      "20", "21","22","23", "24", "25", "26", "27", "28", "29",
                      "30", "31"
                      ]
        else:
            days:list = [str((start.day)+1)]+[str(i+1) for i in range(start.day, dt.date.today().day)]

        for m in months:
            for d in days:
                tweetCriteria = got.manager.TweetCriteria().setQuerySearch(self.company) \
                    .setSince("2020" + "-" + m + "-" + d) \
                    .setUntil("2020" + "-" + m + "-" + str(int(d) + 1)) \
                    .setTopTweets(False) \
                    .setMaxTweets(150) \
                    .setLang('en')
                tweets = got.manager.TweetManager.getTweets(tweetCriteria)
                for tt in tweets:
                    df = df.append({'date': tt.date, 'text': tt.text}, ignore_index=True)
                    #print(tt.text)

        df.to_csv("data/google_tweets_.csv", index=False)
        self.file.close()
        self.update_start()



if __name__ == '__main__':

    t = Tweets()
    #t.get_tweets()
    t.update_tweet_hist()
