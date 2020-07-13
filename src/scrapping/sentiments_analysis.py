from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
import regex as re
from spellchecker import SpellChecker


class Sentimend_analysis():
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tweets =  list(pd.read_csv("data/clean_tweets.csv").values)


    def penn_to_wn(self, tag):

        if tag.startswith('J'): #check nltk.pos_tag documentation
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    def reduce_lengthening(self,text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)

    def specll_check(self,text):
        spell = SpellChecker()
        text = spell.split_words(text)

        return " ".join([spell.correction(word) for word in text])
    def clean_text(self, tweet):
        tweet = tweet.replace("<br />", " ")
        tweet = self.reduce_lengthening(tweet)
        tweet = self.specll_check(tweet)

        return tweet


    def senti_polarity(self, tweet):
        #nltk.download()
        #clean the text
        #tweet = self.clean_text(tweet)
        #convert the tweets text to sentenses_tokens
        self.positive_tweets = 0
        self.negative_tweets= 0
        raw_sentences = sent_tokenize(tweet)
        print("tokenization finished")
        for raw_sentence in raw_sentences:
            raw_sentence = self.reduce_lengthening(raw_sentence)
            #raw_sentence = self.specll_check(raw_sentence)
            print(raw_sentence)
            #foreach sentense we apply wortokenization and call pot tag to get the grammatical nature of the words
            tagged_sentence = pos_tag(word_tokenize(raw_sentence))
            for word, tag in tagged_sentence:

                wn_tag = self.penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                    continue


                #convert each word to its base form
                lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue
                #check if the word exits in the wordnet database
                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())

                sentiment = swn_synset.pos_score() - swn_synset.neg_score()
                if sentiment >= 0:
                    self.positive_tweets += sentiment
                else:
                    self.negative_tweets += sentiment
        print("one tweet finished")

    def tweets_analytics(self):
        data = []
        for tweet in self.tweets:
            self.senti_polarity(tweet[1])
            data.append([tweet[0], self.positive_tweets, self.negative_tweets])
        pd.DataFrame(data, columns=["date","positive", "negative"]).to_csv('data/tweets_scores.csv', index=False)

if __name__ == "__main__":
    senti = Sentimend_analysis()
    senti.tweets_analytics()


