# QuestionParser
My Final Year Project

# Features
### Preprocess Question Input
It does not assume question is of standard input, i.e., free of typo and case sensitive.

Uses [language-check](https://github.com/myint/language-check) for correcting any typo, and
[truecaser](https://github.com/nreimers/truecaser) (converted to Python 3, and wrapped it with OOP class) to convert any caseless / case insensitive word to case sensitive one.

### Extract Linguistic Features
For understanding the question as far as possible, the following features are extracted:
#####  By directly using [spaCy](https://github.com/explosion/spaCy):
  * Noun chunks in the question, from [spaCy documentation](https://spacy.io/api/doc#noun_chunks), 'A base noun phrase, or "NP chunk", is a noun phrase that does not permit other NPs to be nested within it – so no NP-level coordination, no prepositional phrases, and no relative clauses.' 
  * Syntactic dependency of every word in the question
  * Entities in the question, I categorise the [18 entities](https://spacy.io/api/annotation#named-entities) into `PER`, `LOC`, `OBJ`, `TEM`, `NUM`
  * Part of Speech (POS) tag of the first word and the word that has `ROOT` dependency
##### By tweaking spaCy:
  * Noun phrases in the question, any phrase in the `subtree` of words that have any `dep_` that is of `SUBJ`, `OBJT`, `NOUN`, `PREP` (see `constant.py` for my Enum classes). It should be more comprehensive than `noun_chunks`.
  * Neck of the question and its label, second <b>essential</b> element of the question. The neck can be a word, a noun phrase, or a noun chunk, and its label can be the word's dependency, 'np', 'nc' respectively. I skip any prepositional phrase despite it's included in noun phrases, because I feel prepositonal phrases can be moved around any where in the question.
  * Structure of the question, dependency (or noun phrase) order of segmentations of the question. It is shorter than dependency list as some words are marked as noun phrase.
##### By exploring machine learning algorithms:
  * Training data of more than 5000 questions labeled with question type (in line with TREC labeling: ABBR, DESC, LOC, HUM, NUM, ENTY). See `corpus/wh_raw_processed.csv` for the data.
  * Main predictors: `Head`,`Head_POS`, `Neck_label`, `PER` (if has entity in `PER` category), `LOC`, `OBJ`, `TEM`, `NUM`, `ROOT_POS`
  * To predict: `Class` (type of the question)
  * Algorithms used: multinomial logistic regression, support vector machine. See `predict_qn_type.py` for more details.
  * Use `build_train_data.py` to build your own training data and compare the accuracy of different models

Method | Train Data Prediction Accuracy | Test Data Prediction Accuracy 
------------ | ------------ | -------------
Multinomial Logistic Regression | 63% | 65%
Support Vector Machine | 63% | 63%

# To Use
1. Git clone the repo
2. Use [Python 3 venv](https://docs.python.org/3/library/venv.html)
3. Install relevant packages:
```
pip install -r requirements.txt
```
4. Install [spaCy English model](https://spacy.io/usage/models)
```
python -m spacy download en
```
5. Download [english_distributions.obj.zip](https://github.com/nreimers/truecaser/releases), and add the unziped file to your directory
6. Example:
```
from questionparser import QuestionParser
question = "Whar is a question parser?"
qp = QuestionParser(question)
qp.preprocess()
print(qp.get_type()) #'DESC'
```


