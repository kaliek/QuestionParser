import spacy
import language_check
import nltk
import pandas
from constant import *
from truecaser import TrueCaser
from predict_qn_type import get_predict_data, multinomial_regression#, support_vector_machine

class QuestionParser():
    nlp = spacy.load("en")
    tool = language_check.LanguageTool('en-US')
    OBJ_PATH = "distributions.obj"
    truecaser = TrueCaser(OBJ_PATH)

    def __init__(self, question):
        self.question = question
        self.question_head = None
        self.question_doc = None
        self.words = dict((k, "") for k in ['head', 'root', 'neck'])
        self.syntax = {}
        self.phrases = dict((k, []) for k in ['sbjt', 'objt', 'prep'])
        self.entity = dict((k, []) for k in ['per', 'loc', 'obj', 'tem', 'num'])
        self.predict_dta = None
        self.type = ""

    def parse(self):
        self.preprocess()
        self.extract_all()

    def preprocess(self):
        self.correct_sentence()
        self.try_truecaser()
        self.question_doc = self.nlp(self.question)
        self.question_head = self.question_doc[0].text

    def extract_all(self):
        self.extract_syntax()
        self.extract_entity()
        self.extract_words()
        self.extract_predict_dta()
        self.extract_type()

    ####### Machine Leanring Methods for Question Type Prediction #######
    def extract_type(self):
        X_train, y_train, X_predict = get_predict_data(self.predict_dta)
        self.type = multinomial_regression(X_train, y_train, X_predict)

    def extract_predict_dta(self):
        qdata_frame = [{
            'Head': self.question_head,
            'Neck_Label': self.get_word('neck'),
            'PER': self.has_entity('per'),
            'LOC': self.has_entity('loc'),
            'OBJ': self.has_entity('obj'),
            'TEM': self.has_entity('tem'),
            'NUM': self.has_entity('num'),
            'Root_POS': self.get_word('root')
        }]
        dta = pandas.DataFrame(qdata_frame)
        self.predict_dta = dta

    ####### Feature Extraction Methods #######
    def extract_details(self):
        for token in self.question_doc:
            print(token.text, token.lemma_, token.tag_, token.ent_type_, token.dep_, token.head)

    def extract_syntax(self):
        self.iterate_sbjt()
        self.iterate_prep()
        self.iterate_objt()
        self.iterate_others()

    def iterate_sbjt(self):
        for i in range(len(self.question_doc)):
            token = self.question_doc[i]
            if SUBJ.has_value(token.dep_):
                sbjt_list = [w.text_with_ws.strip() for w in token.subtree]
                self.label(i, token.text, sbjt_list, 'sbjt')
                self.phrases['sbjt'].append(" ".join(sbjt_list))

    def iterate_prep(self):
        for i in range(len(self.question_doc)):
            token = self.question_doc[i]
            if PREP.has_value(token.dep_):
                prep_list = [tok.orth_ for tok in token.subtree]
                # prep_list = [w.text_with_ws.strip() for w in token.subtree]
                self.label(i, token.text, prep_list, 'prep')
                self.phrases['prep'].append(" ".join(prep_list))

    def iterate_objt(self):
        for i in range(len(self.question_doc)):
            token = self.question_doc[i]
            if OBJT.has_value(token.dep_):
                objt_list = [w.text_with_ws.strip() for w in token.subtree]
                self.label(i, token.text, objt_list, 'objt')
                self.phrases['objt'].append(" ".join(objt_list))

    def iterate_others(self):
        for i in range(len(self.question_doc)):
            token = self.question_doc[i]
            if self.syntax.get(i) is None:
                self.syntax[i] = token.dep_

    def label(self, i, text, l, string):
        index = l.index(text)
        for j in range(index+1):
            if self.syntax.get(i-j) is None:
                self.syntax[i-j] = string
        for j in range(1, len(l)-index):
            if self.syntax.get(i+j) is None:
                self.syntax[i+j] = string

    def extract_entity(self):
        for ent in self.question_doc.ents:
            if PER.has_value(ent.label_):
                self.entity['per'].append(ent.text)
            elif LOC.has_value(ent.label_):
                self.entity['loc'].append(ent.text)
            elif OBJ.has_value(ent.label_):
                self.entity['obj'].append(ent.text)
            elif TEM.has_value(ent.label_):
                self.entity['tem'].append(ent.text)
            elif NUM.has_value(ent.label_):
                self.entity['num'].append(ent.text)

    # Neck means the phrase/word closest to first word in the question
    def extract_words(self):
        for t in self.question_doc:
            if ROOT.has_value(t.dep_):
                self.words['root'] = t.tag_
        self.words['head'] = self.question_doc[0].tag_
        self.words['neck'] = self.syntax[1]

    ####### Preprocess Methods #######
    def correct_sentence(self):
        print("correcting sentence: ")
        matches = self.tool.check(self.question)
        self.question = language_check.correct(self.question, matches)
        print(self.question)

    def try_truecaser(self):
        print("trying truecase: ")
        tokens = nltk.word_tokenize(self.question)
        tokens = [token.lower() for token in tokens]
        self.question = " ".join(self.truecaser.getTrueCase(tokens))
        print(self.question)

    ####### Getter Methods #######
    def get_head(self):
        return self.question_head

    def get_word(self, string):
        return self.words[string]

    def get_phrase(self, string):
        print(self.phrases)
        return self.phrases[string]

    def get_syntax(self):
        syntax = []
        for (key, value) in sorted(self.syntax.items()):
            if syntax:
                if syntax[len(syntax) - 1] != value:
                    syntax.append(value)
            else:
                syntax.append(value)
        return syntax

    def get_entity(self, string):
        return self.entity[string]

    def get_type(self):
        return self.type

    def has_entity(self, string):
        return 1 if self.get_entity(string) else 0

    def string(self, l):
        return "|".join(l)

# def main():
#     question = "What on the earth are the wonders of the stupid china"
#     qpp = QuestionParser(question)
#     qpp.parse()
#     qpp.extract_details()
#     print(qpp.get_phrase('sbjt'))
#     print(qpp.get_type())
#     print(qpp.get_syntax())

# def run():
#     main()

# if __name__ == "__main__":
#     run()
