source ~/.virtualenvs/parlai/bin/activate
cd ~/beto/ParlAI
python projects/image_chat/server_with_style_chinese.py -mf models:image_chat/transresnet_multimodal/model --response_template_path projects/image_chat/ner/
