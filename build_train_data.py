import os
import csv
import pandas
from sklearn.svm import LinearSVC
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from questionparser import QuestionParser

CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
def compare_model():
    wh_data = pandas.read_csv(os.path.join(CORPUS_DIR, 'all_raw_2.csv'))
    labels = wh_data.pop('Class')
    wh_data.pop('Question')
    wh_data = train_data_matrix(wh_data)
    train_x, test_x, train_y, test_y = train_test_split(wh_data, labels, train_size=0.8)
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
    print("train accuracy: {}".format(metrics.accuracy_score(train_y, model.predict(train_x))))
    print("test accuracy: {}".format(metrics.accuracy_score(test_y, model.predict(test_x))))
    lin = LinearSVC()
    svm_model = lin.fit(train_x, train_y)
    print("svm train accuracy: {}".format(metrics.accuracy_score(train_y, svm_model.predict(train_x))))
    print("svm test accuracy: {}".format(metrics.accuracy_score(test_y, svm_model.predict(test_x))))

def build_data():
    original = ["Question", "Class", "Head"]
    pos_dep = ["Head_POS", "Neck_Label", "Root_POS", "Syntax"]
    entity = ["PER", "LOC", "OBJ", "TEM", "NUM"]
    original.extend(pos_dep)
    original.extend(entity)
    header = original
    with open(os.path.join(CORPUS_DIR, 'all_raw.csv'), "r") as train, open(os.path.join(CORPUS_DIR, 'all_raw_2.csv'), "w") as feature:
        reader = csv.reader(train)
        next(reader, None)
        writer = csv.writer(feature)
        writer.writerow(header)
        for line in reader:
            question = line[0]
            label = line[1]
            print(question)
            qpp = QuestionParser(question)
            qpp.parse()
            result = [question, label, qpp.get_head(),
                qpp.get_word('head'), qpp.get_word('neck'), qpp.get_word('root'), " ".join(qpp.get_syntax()),
                qpp.has_entity('per'), qpp.has_entity('loc'), qpp.has_entity('obj'), qpp.has_entity('tem'), qpp.has_entity('num')]
            print(result)
            writer.writerow(result)

def train_data_matrix(X_train):
    X_train = pandas.get_dummies(X_train)
    X_train_columns = list(X_train.columns)

    trans_data_train = {}
    for col in X_train_columns:
        if col not in X_train:
            trans_data_train[col] = [0 for i in range(len(X_train.index))]
        else:
            trans_data_train[col] = list(X_train[col])
    XT_train = pandas.DataFrame(trans_data_train)
    XT_train = csr_matrix(XT_train)
    return XT_train

if __name__ == "__main__":
    # build_data()
    compare_model()
