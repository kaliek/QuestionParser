import os
import requests
import wikipedia
from questionparser import QuestionParser


class SmartAnswer(QuestionParser):
    def __init__(self, question):
        super().__init__(question)
        super().parse()
        self.loc_answer = []
        self.hum_answer = ""
        self.wiki_answer = []

    # Check if it's a LOC question and has 'LOC' entity
    # if yes: get locaion coordinantes
    def is_loc_answer(self):
        if self.get_type() == 'LOC':
            if self.has_entity('loc'):
                result = get_lat_lng(" ".join(self.get_entity('loc')))
                if result:
                    self.loc_answer = result
            elif self.get_phrase('sbjt'):
                result = get_lat_lng(" ".join(self.get_phrase('sbjt')))
                if result:
                    self.loc_answer = result
        return self.loc_answer

    # Check if it's a 'HUM' question, and has 'PER' entity or noun subject
    # If yes, return one-sentence wiki result
    def is_hum_answer(self):
        if self.get_type() == 'HUM':
            if self.has_entity('per'):
                self.hum_answer = get_wiki_one_sentence(" ".join(self.get_entity('per')))
            elif self.get_phrase('sbjt'):
                self.hum_answer = get_wiki_one_sentence(" ".join(self.get_phrase('sbjt')))
        return self.hum_answer

    # Check if it has any subject or object
    # if yes, get wiki summary
    def is_wiki_answer(self):
        if self.get_phrase('sbjt'):
            result = get_wiki_summary(self.get_phrase('sbjt')[0])
            if result:
                self.wiki_answer.append(result)
        else:
            obj_list = self.get_phrase('objt')
            for obj in obj_list:
                result = get_wiki_summary(obj)
                if result:
                    self.wiki_answer.append(result)
        return self.wiki_answer

# Get location coordinates for loc entity
def get_lat_lng(loc):
    api = ""
    try:
        with open('googlemap_api.txt', 'r') as f:
            api = f.readline().strip()
    except IOError:
        print("No API file. Please create a token.txt with your token in the first line.")
        os.sys.exit()
    response = requests.get("https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}".format(loc, api))
    result = []
    if response.json()["status"] != "ZERO_RESULTS":
        location = response.json()["results"][0]["geometry"]["location"]
        result = [location['lat'], location['lng']]
    return result

# Get one-sentence wili result if obj exists
def get_wiki_one_sentence(obj):
    try:
        raw = wikipedia.page(obj)
    except:
        raw = None
    result = ""
    if raw:
        summ = wikipedia.summary(obj, sentences=1)
        if result:
            result = summ + "\n" + raw.url
    return result

# Get wiki summary if obj exists
def get_wiki_summary(obj):
    try:
        raw = wikipedia.page(obj)
    except:
        raw = None
    result = ""
    if raw:
        summ = wikipedia.summary(obj)
        result = summ + "\n" + raw.url
    return result


# if __name__ == "__main__":
#     question_list = [
#         ""
#     ]
#     for q in question_list:
#         sans = SmartAnswer(q)
#         print(sans.is_loc_answer())
#         print(sans.is_hum_answer())
#         print(sans.is_wiki_answer())
