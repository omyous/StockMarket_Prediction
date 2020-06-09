#import dependencies
from pathlib import Path
import datetime as dt
import pandas_datareader as web


class Stock_data():
    def __init__(self):
        #check if the file that contains the last scrapping data exists
        self.start_path = Path("data/start_file.txt")

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

        #self._raw_data = self.get_data()
        #self.update_start()

    #save the default date in the file
    def save_default_date(self):
        self.start = dt.date(2019, 1, 1)
        self.file = open(self.start_path, 'w+', encoding="utf-8")
        self.file.write(self.start)
        self.file.close()

    #get the data from self.satrt to self.end
    def get_data(self, ticker ="AAPL"):
        return web.data.DataReader(ticker, 'yahoo',self.start, self.end)


    #after the scrapping, we have to update the start date to get the test data afterwhile
    def update_start(self):
        self.file = open(self.start_path, "w+")
        self.file.write(str(self.end))
        self.file.close()
    #define a getter
    @property
    def raw_data(self):
        return self._raw_data

    #define a setteur
    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value


if __name__ == '__main__':
    s = Stock_data()
    s.update_start()




















"""import yfinance
    data=yfinance.download(tickers="GOOGL",start=s.start, end="2020-06-08")
    data.to_csv(path_or_buf="data/google_stock.csv",index=False) print(data)
    
   def get_yahoo_ticker_data(ticker):
       res = requests.get('https://finance.yahoo.com/quote/' + ticker + '/history')
       yahoo_cookie = res.cookies['B']
       yahoo_crumb = None
       pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')
       for line in res.text.splitlines():
           m = pattern.match(line)
           if m is not None:
               yahoo_crumb = m.groupdict()['crumb']
       cookie_tuple = yahoo_cookie, yahoo_crumb

       current_date = int(time.time())
       url_kwargs = {'symbol': ticker, 'timestamp_end': current_date,
                     'crumb': cookie_tuple[1]}
       url_price = 'https://query1.finance.yahoo.com/v7/finance/download/' \
                   '{symbol}?period1=0&period2={timestamp_end}&interval=1d&events=history' \
                   '&crumb={crumb}'.format(**url_kwargs)

       response = requests.get(url_price, cookies={'B': cookie_tuple[0]})

       return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


   print(get_yahoo_ticker_data(ticker='GOOGL'))    
    
    """






