import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from newspaper import Article, article
import re


class Recapper():

    def __init__(self, url):
        try:
            pattern = re.compile("^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$")
            if not pattern.match(url):
                print(f"{url} is not a valid url")
            self.url = url
            self.article = Article(self.url)
            self.article.download()
            self.article.parse()
            self.author = self.article.authors
            self.oneline = self.article.summary
            self.text = self.article.text.replace("\n", "")
            if self.article.meta_lang == 'en':
                import en_core_web_sm
                self.model = en_core_web_sm.load()
            elif self.article.meta_lang == 'it':
                import it_core_news_sm
                self.model = it_core_news_sm.load()
            elif self.article.meta_lang == 'fr':
                import fr_core_news_sm
                self.model = fr_core_news_sm.load()
            elif self.article.meta_lang == 'es':
                import es_core_news_sm
                self.model = es_core_news_sm.load()
            elif self.article.meta_lang == 'pt':
                import pt_core_news_sm
                self.model = pt_core_news_sm.load()
            else:
                print("language not supported")
            self.data = []
            self.vectorizer = TfidfVectorizer(strip_accents='unicode')
        except article.ArticleException:
            print(f"The url {url} is not supported, please write to email@towrite.it for further help")
            self.valid=False

    def process(self):
        try:
            data = []
            lemmatized = []
            for s in self.text.split('.'):
                self.data.append(s)
                data.append(self.model(s))
            for sentence in data:
                words = []
                for token in sentence:
                    if (
                            token.pos_ == 'NOUN' or token.pos_ == 'NUM' or token.pos_ == 'PROPN' or token.pos_ == 'VERB' or token.pos_ == 'X') and not token.is_stop:
                        words.append(token.lemma_)
                lemmatized.append(" ".join(words))
            lemmatized = pd.DataFrame(lemmatized, columns=["sentence"])
            fq = self.vectorizer.fit_transform(lemmatized["sentence"])
            self.frequencies = pd.DataFrame(fq.toarray())
            self.recap = pd.DataFrame(
                {"Sentence": self.data, "Score": self.frequencies.mean(axis=1)})
            self.recap["Rank"] = self.recap["Score"].rank(method="first", ascending=False)
            self.tfidf_sorting = np.argsort(fq.toarray()).flatten()[::-1]
        except (ValueError,AttributeError):
            print(
                f"The text at {self.url} could not be processed, please write to mail@email.com to have further help")

    def summarize(self, perc=0.3):
        try:
            n_sentences = round(len(self.recap) * perc)
            self.summary = "\n".join(
                (self.recap.loc[
                    (self.recap['Rank'] <= n_sentences) | (self.recap['Rank'] == self.recap["Rank"].iloc[0])])[
                    "Sentence"])
            return self.summary
        except (ValueError,AttributeError):
            message="Some errors occured during the processing and or the creation of the object"
            print(message)
            return message

    def get_info(self, n=5, recap=False):
        feature_array = np.array(self.vectorizer.get_feature_names())
        self.top_words = feature_array[self.tfidf_sorting][:n]
        print(f"Most used words are: {self.top_words}".replace("[", "").replace("]", ""))
        print(f"The summary is the {round(len(self.summary) / len(self.text) * 100)}% of the original")
        print(self.recap) if recap == True else False

    def dependencies_and_licence(self):
        message = "The current software uses the spacy and newspaper3k libraries, both under MIT Licence\n"+"newspaper3k -> https://github.com/codelucas/newspaper/blob/master/LICENSE\n"+"spacy -> https://github.com/explosion/spaCy/blob/master/LICENSE"
        return message
