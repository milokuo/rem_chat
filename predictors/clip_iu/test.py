from concept_extractor import ConceptExtractor
from clip_predictor import ClipPredictor
from senti_net import SentiNet
from gpt_generator import GptGenerator

from sentence_transformers import SentenceTransformer
# _sentence_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if __name__ == '__main__':
    ce = ConceptExtractor()
    # ce.search_causes('marriage', 3)

    cp = ClipPredictor()
    # url = "http://140.112.95.5:41732/src/uploads/138_20230103_121645.jpg"

    # labels = ['happy', 'sad', 'angry', 'anxiety']
    # text_candidates = [f'a photo of {label} people' for label in labels]
    # cp.predict(url, text_candidates)

    sa = SentiNet()
    # sa.get_senti('I love swimming.')    

    _gpt = GptGenerator()

    url = "http://140.112.95.5:41732/src/uploads/138_20230103_121645.jpg"

    user_input = 'I feel uncomfortable since I have had a bad headache.'
    
    for i in range(100):
        # user_input = input('user: \t')
        
        _senti = sa.get_senti(user_input)
        print(f'user sentiment: {_senti}')

        _user_key = _gpt.inference_3p5(f'use one word to describe the key information that affects user emotion according to his/her sentence: \n{user_input}')
        print(f'user key: {_user_key}')

        _causes = ce.search_causes(_user_key, 3)
        print(print(f'causes: {_causes}'))
        _causes_str = ','.join(_causes)
        if _causes == None or len(_causes) == 0:
            _causes_str = _gpt.inference_3p5(f'provide me some causes of {_user_key} in the format of string array')
            print(_causes_str)
        
        _next_key = _gpt.inference_3p5(f'choose the most relevant cause from {_causes_str} that affects user emotion according to this sentence: \n{user_input}')
        print(f'next key: {_next_key}')
        template = f'please reply me from the perspective of {_next_key}'

        gpt_output = _gpt.inference_3p5(','.join([user_input, f'the user seems {_senti}', template]))
        print(f'robot: \t{gpt_output}')

        user_input = input('user: \t')
