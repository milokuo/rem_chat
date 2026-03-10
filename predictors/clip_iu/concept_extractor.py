# -*- coding:utf-8 -*-
import requests

class ConceptExtractor:
    def search_causes(self, end, limit):
        obj = requests.get(f'http://api.conceptnet.io/query?rel=/r/Causes&end=/c/en/{end}&limit={limit}').json()
        # print(obj.keys()) #dict_keys(['view', '@context', '@id', 'edges'])
        # print('edges: {}'.format(len(obj['edges'])))
        causes = []
        for edge in obj['edges']:
            print('{}---{}/{}-->{}'.format(edge['start']['label'], edge['rel']['label'], edge['weight'], edge['end']['label']))
            causes.append(edge['start']['label'])
        return causes


if __name__ == '__main__':
    ce = ConceptExtractor()
    ce.search_causes('marriage', 3)