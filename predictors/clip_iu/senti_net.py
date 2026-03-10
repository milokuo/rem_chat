# -*- coding:utf-8 -*-
from transformers import pipeline


class SentiNet:
    def __init__(self):
        self._sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
    
    def predict(self, user_input):
        senti_output = self._sentiment_analysis(user_input)
        # print(senti_output)
        senti = senti_output[0]['label']
        return senti
    


if __name__ == '__main__':
    sa = SentiNet()
    sa.predict('I love swimming.')