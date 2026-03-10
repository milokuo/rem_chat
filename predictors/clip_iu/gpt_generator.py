# -*- coding:utf-8 -*-
import os
import openai


class GptGenerator:
        prompt_candidate = {
            "reply": "Please provide a response coherent to user content.",
            "comfort": "Please provide an empathetic response to the user.",
            "arousing": "Ask the user to talk more about the <topic> of the photography.",
            "compromise": "Now forget the photography. Just make the conversation following the user\'s chain of thinking."
        }
        essentials = {"scene", "place", "role", "date", "else"}

    def __init__(self):
        # API key 由啟動程式透過 config.py 設定至環境變數
        openai.api_key = os.environ.get('OPENAI_API_KEY', '')

    def inference_3(self, prompt):
        output = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0
            )
        return output['choices'][0]['text'].strip()

    def inference_3p5(self, prompt):
        completion = openai.ChatCompletion.create (
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            )
        return completion['choices'][0]['message']['content'].strip()


if __name__ == '__main__':
    _gpt = GptGenerator()

    for i in range(100):
        user_input = input('user: \t')

        #TODO
        key = input('prompt key:\t')
        template = f'please reply me from the perspective of {key}'

        senti = input('senti:\t')

        gpt_output = _gpt.inference_3p5(','.join([user_input, f'the user seems {senti}', template]))
        print(f'robot: \t{gpt_output}')

    
        

        