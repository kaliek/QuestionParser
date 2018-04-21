import os
import csv
import pandas
from sklearn.svm import LinearSVC
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from questionparser import QuestionParser

CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
def compare_model(train_file, test_file):
    train_data = pandas.read_csv(train_file)
    labels = train_data.pop('Class')
    train_data.pop('Question')
    test_data = pandas.read_csv(test_file)
    test_labels = test_data.pop('Class')
    test_data.pop('Question')

    X_train, X_test = transform_data_matrix(train_data, test_data)
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, labels)
    print("train accuracy: {}".format(metrics.accuracy_score(labels, model.predict(X_train))))
    print("test accuracy: {}".format(metrics.accuracy_score(test_labels, model.predict(X_test))))
    lin = LinearSVC()
    svm_model = lin.fit(X_train, labels)
    print("svm train accuracy: {}".format(metrics.accuracy_score(labels, svm_model.predict(X_train))))
    print("svm test accuracy: {}".format(metrics.accuracy_score(test_labels, svm_model.predict(X_test))))
    # train_x, test_x, train_y, test_y = train_test_split(wh_data, labels, train_size=0.8)
    # model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
    # print("train accuracy: {}".format(metrics.accuracy_score(train_y, model.predict(train_x))))
    # print("test accuracy: {}".format(metrics.accuracy_score(test_y, model.predict(test_x))))
    # lin = LinearSVC()
    # svm_model = lin.fit(train_x, train_y)
    # print("svm train accuracy: {}".format(metrics.accuracy_score(train_y, svm_model.predict(train_x))))
    # print("svm test accuracy: {}".format(metrics.accuracy_score(test_y, svm_model.predict(test_x))))

def build_data(input_file, output_file):
    original = ["Question", "Class", "Head"]
    pos_dep = ["Head_POS", "Neck_Label", "Root_POS", "Syntax"]
    entity = ["PER", "LOC", "OBJ", "TEM", "NUM"]
    original.extend(pos_dep)
    original.extend(entity)
    header = original
    with open(input_file, encoding = "ISO-8859-1") as train, open(output_file, "w") as feature:
        writer = csv.writer(feature)
        writer.writerow(header)
        for line in train:
            line = line.split(":")
            label = line[0]
            question = " ".join(line[1].split()[1:])
            print(question)
            qpp = QuestionParser(question)
            qpp.parse()
            result = [question, label, qpp.get_head(),
                qpp.get_word('head'), qpp.get_word('neck'), qpp.get_word('root'), " ".join(qpp.get_syntax()),
                qpp.has_entity('per'), qpp.has_entity('loc'), qpp.has_entity('obj'), qpp.has_entity('tem'), qpp.has_entity('num')]
            print(result)
            writer.writerow(result)

def add_rating_data(rating_file, training_file):
    with open(rating_file, "r") as train, open(os.path.join(CORPUS_DIR, 'all_corpus_2_copy.csv'), "a") as feature:
        reader = csv.reader(train)
        next(reader, None)
        writer = csv.writer(feature)
        for line in reader:
            rating = line[2]
            if rating == "T":
                question = line[0]
                label = line[1]
                qpp = QuestionParser(question)
                qpp.parse()
                result = [question, label, qpp.get_head(),
                    qpp.get_word('head'), qpp.get_word('neck'), qpp.get_word('root'), " ".join(qpp.get_syntax()),
                    qpp.has_entity('per'), qpp.has_entity('loc'), qpp.has_entity('obj'), qpp.has_entity('tem'), qpp.has_entity('num')]
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

def transform_data_matrix(X_train, X_predict):
    X_train = pandas.get_dummies(X_train)
    X_predict = pandas.get_dummies(X_predict)
    X_train_columns = list(X_train.columns)
    X_predict_columns = list(X_predict.columns)
    X_trans_columns = list(set(X_train_columns + X_predict_columns))

    trans_data_train = {}
    for col in X_trans_columns:
        if col not in X_train:
            trans_data_train[col] = [0 for i in range(len(X_train.index))]
        else:
            trans_data_train[col] = list(X_train[col])
    XT_train = pandas.DataFrame(trans_data_train)
    XT_train = csr_matrix(XT_train)

    trans_data_predict = {}
    for col in X_trans_columns:
        if col not in X_predict:
            trans_data_predict[col] = [0 for i in range(len(X_predict.index))]
        else:
            trans_data_predict[col] = list(X_predict[col])  # KeyError
    XT_predict = pandas.DataFrame(trans_data_predict)
    XT_predict = csr_matrix(XT_predict)
    return XT_train, XT_predict

# if __name__ == "__main__":
#     # input_file = os.path.join(CORPUS_DIR, 'train_5000.label.txt')
#     # output_file = os.path.join(CORPUS_DIR, 'temp_features.csv')
#     # build_data(input_file, output_file)
#     train = os.path.join(CORPUS_DIR, 'train_5500_features.csv')
#     test = os.path.join(CORPUS_DIR, 'TREC_10_features.csv')
#     compare_model(train, test)
#     rating_file = os.path.join(CORPUS_DIR, 'rating.csv')
#     # training_file = 
#     # add_rating_data(rating_file, training_file)
