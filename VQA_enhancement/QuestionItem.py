# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import json
import os


class QuestionItem:
    def __init__(self, idx=-1, category="", label="", question="", embedding=None, similar=None):
        if similar is None:
            similar = []
        self.idx = idx
        self.category = category
        self.label = label
        self.question = question
        self.embedding = embedding
        self.similar = similar
        self.confidence = 0.0

    def getIdx(self):
        return self.idx

    def setIdx(self, idx):
        self.idx = idx

    def getCategory(self):
        return self.category

    def setCategory(self, category):
        self.category = category

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label = label

    def getQuestion(self):
        return self.question

    def setQuestion(self, question):
        self.question = question

    def getEmbedding(self):
        return self.embedding

    def setEmbedding(self, embedding):
        self.embedding = embedding

    def getSimilar(self):
        return self.similar

    def setSimilar(self, similar):
        self.similar = similar

    def getConfidence(self):
        return self.confidence

    def setConfidence(self, confidence):
        self.confidence = confidence

    def toString(self):
        return "QuestionItem{" + \
               "id=" + str(self.idx) + \
               ", category='" + self.category + '\'' + \
               ", label='" + self.label + '\'' + \
               ", question='" + self.question + '\'' + \
               '}'

    def fromJson(self, jdata):
        self.idx = jdata['id']
        self.category = jdata['category']
        self.label = jdata['label']
        self.question = jdata['question']
        self.similar = jdata['similar']


class QuestionSet:
    def __init__(self, path):
        with open(os.path.join(path, "image_labels.json"), 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print("loaded labels: {}".format(labels))

        df = pd.read_csv(os.path.join(path, "list_all.txt"), sep='\t', header=None, engine="python")
        print("loaded Q list with index: \n{}".format(df))

        self.em = np.load(os.path.join(path, "questions_embedding.npy"))
        print("loaded Q embeddings with shape: {}".format(self.em.shape))

        with open(os.path.join(path, "questions.json"), 'r', encoding='utf-8') as f:
            questions = json.load(f)
        # print("loaded Q details: {}".format(questions))
        print(pd.DataFrame(questions))
        self.df_qAll = pd.DataFrame(questions)  # better for sorting

        self.qAll = []
        for item in questions:
            qItem = QuestionItem()
            qItem.fromJson(item)
            qItem.setEmbedding(self.em[qItem.getIdx()])  # (512,)
            self.qAll.append(qItem)

        print("loaded {} questions".format(len(self.qAll)))

        self.qEvent = []
        self.qPlace = []
        self.qRelation = []
        self.qSelected = {}

    def get_questions(self, metadata):
        # reset first
        self.qEvent = []
        self.qPlace = []
        self.qRelation = []
        self.qSelected = {}

        for qItem in self.qAll:
            if qItem.getCategory() == 'event' and qItem.getLabel() == metadata['event']['label']:
                qItem.setConfidence(metadata['event']['confidence'])
                self.qEvent.append(qItem)
            if qItem.getCategory() == 'place' and qItem.getLabel() == metadata['place']['label']:
                qItem.setConfidence(metadata['place']['confidence'])
                self.qPlace.append(qItem)
            if qItem.getCategory() == 'relationship' and qItem.getLabel() == metadata['relationship']['label']:
                qItem.setConfidence(metadata['relationship']['confidence'])
                self.qRelation.append(qItem)

        for cate, qList in zip(['event', 'place', 'relationship'], [self.qEvent, self.qPlace, self.qRelation]):
            print("cate: {}, label: {} {}, # of questions: {}".format(cate,
                                                                      metadata[cate]['confidence'],
                                                                      metadata[cate]['label'],
                                                                      len(qList)))

    def filter_similar(self, q_selected):
        for qSet in [self.qEvent, self.qPlace, self.qRelation]:
            for qItem in qSet:
                if qItem.getIdx() in q_selected.getSimilar():
                    qSet.remove(qItem)
