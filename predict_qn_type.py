from scipy.sparse import csr_matrix
import pandas
from sklearn.model_selection import train_test_split
def support_vector_machine(X_train, y, X_predict):
    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y)
    prediction = lin_clf.predict(X_predict)
    return prediction

def multinomial_regression(X_train, y, X_predict):
    model = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg').fit(X_train, y) 
    prediction = model.predict(X_predict)
    return prediction
       

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
            trans_data_predict[col] =[0 for i in range(len(X_predict.index))] 
        else:
            trans_data_predict[col] = list(X_predict[col])  # KeyError
    XT_predict = pandas.DataFrame(trans_data_predict)
    XT_predict = csr_matrix(XT_predict)
    return XT_train, XT_predict


def c_model():
    wh_data = pandas.read_csv(os.path.join(CORPUS_DIR, 'wh_raw_processed.csv'))
    labels = wh_data.pop('Class')
    wh_data = train_data_matrix(wh_data)
    train_x, test_x, train_y, test_y = train_test_split(wh_data, labels, train_size = 0.7)
    model = linear_model.LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg').fit(train_x, train_y)
    print("train accuracy: {}".format(metrics.accuracy_score(train_y, model.predict(train_x))))
    print("test accuracy: {}".format(metrics.accuracy_score(test_y, model.predict(test_x))))
    lin = LinearSVC()
    svm_model = lin.fit(train_x, train_y)
    print("svm train accuracy: {}".format(metrics.accuracy_score(train_y, svm_model.predict(train_x))))
    print("svm test accuracy: {}".format(metrics.accuracy_score(test_y, svm_model.predict(test_x)))) 