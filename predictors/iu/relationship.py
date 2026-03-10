import os
import json
import operator

class Person(object):

    age_classes = ['young','adult','elder']
    sex_classes = ['female','male']
    
    def __init__(self, age, gender, pose):
        self.meta = {}

        self.person = {}
        self.person['age'] = str(age + 1)
        self.person['gender'] = str(gender + 1)

        self.bbox = {}
        self.bbox['x'] = pose['x']
        self.bbox['y'] = pose['y']
        self.bbox['w'] = pose['w']
        self.bbox['h'] = pose['h']

    def get(self):
        return self.person, self.bbox

class People(object):

    def __init__(self, filenames, ages, genders, poses):
        self.output = {}
        self.output['filenames'] = filenames
        self.output['people'] = {}
        self.output['bboxes'] = {}

        for (filename, age, gender, pose) in zip(filenames, ages, genders, poses):
            person = Person(age, gender, pose)
            people, bboxes = person.get()
            self.output['people'][filename] = people
            self.output['bboxes'][filename] = bboxes

    def getOutput(self):
        return self.output

    def toJSON(self):
        return json.dumps(self.output)

class RelationshipPredictor(object):
    path = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(path, 'labels')

    def __init__(self, people):
        self.people = people
        self.preds = {}

    def predict(self):
        labels_file = os.path.join(self.labels_path, "image_labels.json")
        with open(labels_file) as label_file:
            data = json.load(label_file)
            headers = list(data["relationship"].keys())

        self.preds = [0] * len(headers)
        my_people = list(self.people.values())

        if len(my_people) == 0:
            self.preds[4] = 1 #None
        elif len(my_people) == 1:
            self.preds[5] = 1 #Solo
        elif len(my_people) == 2:
            if my_people[0]["gender"] != my_people[1]["gender"]:
                if abs(int(my_people[0]["age"]) - int(my_people[1]["age"])) < 1:
                    print("tiki")
                    self.preds[0] = 1 #Couple
                else:
                    self.preds[3] = 1 #General
            else:
                self.preds[3] = 1 #General
        else:
            oldest = max(my_people, key=lambda x:x['age'])
            youngest = min(my_people, key=lambda x:x['age'])
            
            if int(oldest["age"]) - int(youngest["age"]) > 1:
                self.preds[1] = 1 #Family
            else:
                self.preds[3] = 1 #General

        output = {}

        for i in range(len(self.preds)):
            output[headers[i]] = float(str('%.1f' % self.preds[i]))

        top = max(output.items(), key=operator.itemgetter(1))

        result = {}
        result["label"] = top[0]
        result["confidence"] = top[1]

        return result