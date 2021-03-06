import os
import pandas
from sklearn.svm import LinearSVC
from sklearn import linear_model
from scipy.sparse import csr_matrix

def support_vector_machine(X_train, y, X_predict):
    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y)
    prediction = lin_clf.predict(X_predict)
    return prediction[0]

def multinomial_regression(X_train, y, X_predict):
    model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y)
    prediction = model.predict(X_predict)
    return prediction[0]

def decision_tree(head, neck_label, root_pos, loc):
    predict_class = ""
    if head == 'who':
        predict_class = 'HUM'
    else:
        if head == 'how':
            if neck_label == 'aux':
                predict_class = 'DESC'
            else:
                predict_class = 'NUM'
        elif head == 'when':
            predict_class = 'NUM'
        elif head == 'where':
            predict_class = 'LOC'
        else:
            neck_set = ['ROOT', 'advmod', 'aux', 'auxpass', 'ccomp', 'intj', 'npadvmod', 'prep']
            if neck_label in neck_set:
                predict_class = 'DESC'
            else:
                root_set = ['NNP', 'UH', 'VBD', 'VBG']
                if root_pos in root_set:
                    predict_class = 'HUM'
                else:
                    if loc:
                        predict_class = 'LOC'
                    else:
                        predict_class = 'ENTY'
    return predict_class

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

def get_predict_data(predict_dta):
    CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
    wh_all = pandas.read_csv(os.path.join(CORPUS_DIR, 'train_5500_features copy.csv'))
    y_train = wh_all.pop('Class')
    wh_all.pop('Question')
    wh_all = pandas.DataFrame(wh_all)
    X_train, X_predict = transform_data_matrix(wh_all, predict_dta)
    return X_train, y_train, X_predict
