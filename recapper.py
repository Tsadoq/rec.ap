import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class Recapper():

    def __init__(self, text, model):

        self.text = text.replace("\n", "")
        self.model = model
        self.data = []
        self.vectorizer = TfidfVectorizer(strip_accents='unicode')

    def process(self):
        data = []
        lemmatized = []
        for s in self.text.split('.'):
            self.data.append(s)
            data.append(self.model(s))
        for sentence in data:
            words = []
            for token in sentence:
                if token.pos_ == 'NOUN' or token.pos_ == 'NUM' or token.pos_ == 'PROPN' or token.pos_ == 'VERB' or token.pos_ == 'X':
                    words.append(token.lemma_)
            lemmatized.append(" ".join(words))
        lemmatized = pd.DataFrame(lemmatized, columns=["sentence"])
        self.frequencies = pd.DataFrame(self.vectorizer.fit_transform(lemmatized["sentence"]).toarray())
        self.recap = pd.DataFrame(
            {"Sentence": self.data, "Score": self.frequencies.mean(axis=1)})
        self.recap["Rank"] = self.recap["Score"].rank(method="first",ascending=False)

    def summarize(self,perc=0.3):
        n_sentences = round(len(self.recap)*perc)
        summary = "\n".join((self.recap.loc[self.recap['Rank'] <= n_sentences])["Sentence"])
        return summary

    def get_info(self):
        print(self.recap)


f = open("text_it.txt", "r")
text = f.read()
import it_core_news_sm

r = Recapper(text, it_core_news_sm.load())
r.process()
print(r.summarize())

r.get_info()
