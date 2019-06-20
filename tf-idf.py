from gensim import corpora
from gensim import models
import spacy
import math

nlp = spacy.load('en_core_web_sm')

class tfidf:
    def __init__(self, document):
        self.doc = [[word.lower_ for word in nlp(line)] for line in document]
        self.dic = corpora.Dictionary(self.doc)
        self.corpus = list(map(self.dic.doc2bow, self.doc))
        self.model = models.TfidfModel(self.corpus, wglobal=self._new_idf, normalize=False)
        self.tf_idf = [[(self.dic[result[0]], result[1]) for result in doc]for doc in self.model[self.corpus]]

    def _new_idf(self, docfreq, totaldocs, log_base=2.0, add=1.0):
        return add + math.log(1.0 * totaldocs / docfreq, log_base)

    def get_score(self, sentence):
        sentence = [[word.lower_ for word in nlp(sentence)]]
        query = list(map(self.dic.doc2bow, sentence))
        return [(self.dic[result[0]], result[1]) for doc in self.model[query] for result in doc]

    def get_topk(self, k):
        return [sorted(doc, key=lambda x:x[1])[:min(len(doc), k)] for doc in self.tf_idf]


if __name__ == '__main__':
    model = tfidf(['a b c a', 'c b c a', 'b b a', 'a c c', 'c b a'])
    print(model.tf_idf)
    print(model.get_score('a c'))