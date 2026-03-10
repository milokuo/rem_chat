import os
import openai

class GptGenerator:
    def __init__(self):
        # API key 由啟動程式（chat_engine.py）透過 config.py 設定至環境變數
        openai.api_key = os.environ.get('OPENAI_API_KEY', '')
        self.messages = [{"role": "system", "content": "You are a companion robot that provides empathetic response to the user. Your task is chatting with the user according to the content of an image. You can propose questions to help the user recall the memomery of that day, and actively response to the user's expressions if they try to tell you the story. But it need to be memtioned that the reponse should be one or two sentences to make the conversation fluent."}]

    def get_response(self, prompt):
        output = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0
            )
        return output['choices'][0]['text'].strip()

    def get_response_turbo(self):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=self.messages,
            temperature=0
            )
        return completion['choices'][0]['message']['content'].strip()

    def get_response_vlm(self):
        completion = openai.ChatCompletion.create(
            model="gpt-4-vision-preview", 
            messages=self.messages,
            temperature=0
            )
        return completion['choices'][0]['message']['content'].strip()

    def reset_system(self, system_config):
        self.messages = []
        self.messages.append({"role": "system", "content": system_config})

    def update_robot(self, robot_reply):
        self.messages.append({"role": "assistant", "content": robot_reply})
    
    def update_user(self, utterance):
        self.messages.append({"role": "user", "content": utterance})

    def reset_history(self):
        self.messages = [{"role": "system", "content": "You are a companion robot that provides empathetic response to the user. Your task is chatting with the user according to the content of an image. You can propose questions to help the user recall the memomery of that day, and actively response to the user's expressions if they try to tell you the story. But it need to be memtioned that the reponse should be one or two sentences to make the conversation fluent."}]
    
    def update_metadata(self, metadata):
        self.messages[0]["content"] += f"This is a photo of {metadata['event']} taken at the place of {metadata['place']}, the people on the photo are {metadata['relationship']}."
