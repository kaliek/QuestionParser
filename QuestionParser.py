import spacy
import language_check
import nltk
from nltk.metrics import edit_distance
import pandas
import csv
import random
import os
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from constant import *
from Truecaser import *
from predict_qn_type import train_data_matrix, predict_data_matrix

class QuestionParser():
    nlp = spacy.load("en")
    tool = language_check.LanguageTool('en-US')
    CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
    wh_all = pandas.read_csv(os.path.join(CORPUS_DIR, 'wh_raw_processed.csv'))
    y_train = wh_all.pop('Class')
    X_train = train_data_matrix(wh_all)
    multinormial = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg').fit(X_train, y_train)
    svm = LinearSVC().fit(X_train, y_train)

    def __init__(self, question):
        self.question = question
        self.question_head = None
        self.question_list = None
        self.question_doc = None
        self.entity = []
        self.entity_label = []
        self.dep_elements = []
        self.root_pos = ""
        self.noun_phrases = []
        self.noun_chunks = []
        self.neck = ""
        self.neck_label = ""
        self.has_per = False
        self.has_loc = False
        self.has_obj = False
        self.has_tem = False
        self.has_num = False
        self.type = ""
        self.structure = []

        self.loc_entity = []


    def preprocess(self):
        self.correct_sentence()
        self.try_truecaser()
        self.question_list = [t.text for t in self.nlp(self.question)]
        self.question_head = self.question_list[0]
        self.question_doc = self.nlp(self.question)
        self.extract_dep()
        self.extract_noun_phrase()
        self.extract_noun_chunk()
        self.extract_entity()
        self.extract_neck()
        self.extract_structure()
        self.predict_type()

    """Machine learning algo to predict the question type"""
    def predict_type(self):
        qdata_frame = [{
            'Head': self.question_head,
            'Neck_Label': self.neck_label,
            'PER': self.get_has('per'),
            'LOC': self.get_has('loc'),
            'OBJ': self.get_has('obj'),
            'TEM': self.get_has('tem'),
            'NUM': self.get_has('num'),
            'Root_POS': self.root_pos
        }]
        dta = pandas.DataFrame(qdata_frame)
        X_predict = predict_data_matrix(self.wh_all, dta)
        self.type = self.multinormial.predict(X_predict)[0]


    """Parsing questions in different ways"""
    def extract_details(self):
        for token in self.question_doc:
            print(token.text, token.lemma_, token.tag_, token.ent_type_, token.dep_, token.head)

    def extract_dep(self):
        for t in self.question_doc:
            if ROOT.has_value(t.dep_): self.root_pos = t.tag_
            self.dep_elements.append(t.dep_)

    def iter_nps(self):
        for word in self.question_doc:
            if SUBJ.has_value(word.dep_) or OBJT.has_value(word.dep_) or NOUN.has_value(word.dep_) or PREP.has_value(word.dep_):
                yield word.subtree
    
    def extract_noun_phrase(self):
        for st in self.iter_nps():
            self.noun_phrases.append(" ".join(t.text for t in st))

    def extract_noun_chunk(self):
        for nc in self.question_doc.noun_chunks:
            self.noun_chunks.append(nc.text)
    
    def extract_entity(self):
        for ent in self.question_doc.ents:
            self.entity.append(ent.text)
            self.entity_label.append(ent.label_)
            if PER.has_value(ent.label_):
                self.has_per = True
            elif LOC.has_value(ent.label_):
                self.has_loc = True
                self.loc_entity.append(ent.text)
            elif OBJ.has_value(ent.label_):
                self.has_obj = True
            elif TEM.has_value(ent.label_):
                self.has_tem = True
            elif NUM.has_value(ent.label_):
                self.has_num = True

    # Neck means the phrase/word closest to first word in the question
    def extract_neck(self):
        if self.question_doc:
            self.neck = self.question_doc[1].text
            self.neck_label = self.dep_elements[1]
            for np in self.noun_phrases:
                if self.neck in np.split():
                    if PREP.has_value(self.neck_label): continue
                    else:
                        self.neck = np
                        self.neck_label = "np"
                        break
            if self.neck == self.question_doc[1].text:
                if self.noun_chunks: #assume noun_chunks does not contain prep phrase
                    if self.neck in self.noun_chunks[0].split():
                        self.neck = self.noun_chunks[0]
                        self.neck_label = "nc"

    # Break the question into different parts
    def extract_structure(self):
        structure = []
        nps = self.noun_phrases
        length_doc = len(self.question_doc)
        i = 0
        while i < length_doc:
            if not nps:
                structure.append(self.question_doc[i].dep_)
                i += 1
            else: 
                first_list = nps[0].split()
                length = len(first_list)
                if self.question_doc[i].text in first_list:
                    structure.append('np')
                    i += length
                    nps.pop(0)
                else: 
                    structure.append(self.question_doc[i].dep_)
                    i += 1
        self.structure = structure
    

    """Correct the typo and caseless words if the question contains"""
    def correct_sentence(self):
        print("correcting sentence: ")
        matches = self.tool.check(self.question)
        self.question = language_check.correct(self.question, matches)
        print(self.question)

    def try_truecaser(self):
        print("trying truecase: ")
        tokens = nltk.word_tokenize(self.question)
        tokens = [token.lower() for token in tokens]
        self.question = " ".join(getTrueCase(tokens))
        print(self.question)
    
    def get_head(self):
        return self.question_head
    
    def get_neck(self):
        return self.neck
    
    def get_neck_label(self):
        return self.neck_label
    
    def get_dep(self):
        return self.dep_elements

    def get_root_pos(self):
        return self.root_pos
    
    def get_noun_phrase(self):
        return self.noun_phrases

    def get_noun_chunk(self):
        return self.noun_chunk
    
    def get_entity(self):
        return self.entity
    
    def get_entity_label(self):
        return self.entity_label

    def get_structure(self):
        return self.structure

    def get_type(self):
        return self.type

    def get_loc_entity(self):
        return self.loc_entity

    def get_has(self, string):
        return 1 if getattr(self, 'has_' + string) else 0
    
    def string(self, l):
        return "|".join(l)



def main():
    question = "Hoq did serfdom develop in and then leave the stupid russia ?"
    qpp = QuestionParser(question)
    qpp.preprocess()
    print(qpp.get_type())


def run():
    main()


if __name__ == "__main__":
    run()