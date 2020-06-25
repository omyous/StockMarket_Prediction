from stocknews import StockNews
from src.scrapping.data.keys import *
import regex as re
import pandas as pd
if __name__=="__main__":
   def reduce_lengthening(text):
      pattern = re.compile(r"(.)\1{2,}")
      return pattern.sub(r"\1\1", text)

   from spellchecker import SpellChecker

   def specll_check(words):
         spell = SpellChecker()
         words = spell.split_words(words)

         return " ".join([spell.correction(word) for word in words])


   str="Disssruption bettween employee DDDoncaste"
   str = reduce_lengthening(str)
   str = specll_check(str)
   print(str)

