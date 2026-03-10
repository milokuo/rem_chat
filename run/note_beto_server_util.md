# start server
cd /opt/lampp
sudo ./lampp start
#test website http://localhost/phpmyadmin/index.php
#if error
        sudo netstat -nap | grep :80
        #find apache2 with 3-4 digit number
        sudo kill <3-4 digit number>
        sudo ./lampp restart


# start chat module
source ~/Xiaobei/parlai_xb/bin/activate
cd ~/Xiaobei/ParlAI

# BlenderBot 1.0
python projects/image_chat/server_bare_blenderbot.py

# BlenderBot 2.0
# BlenderBot 2.0 400m: --model-file zoo:blenderbot2/blenderbot2_400M/model
# BlenderBot 2.0 2.7B (model card): --model-file zoo:blenderbot2/blenderbot2_3B/model
python projects/image_chat/server_updated_blenderbot.py -mf zoo:blenderbot2/blenderbot2_400M/model --search-server 0.0.0.0:8080

# BlenderBot 3.0
# BlenderBot 3  3B: --model-file zoo:bb3/bb3_3B/model 
# BlenderBot 3 30B: --model-file zoo:bb3/bb3_30B/model
python projects/image_chat/server_updated_blenderbot.py -mf zoo:bb3/bb3_3B/model --search-server 0.0.0.0:8080



# Beto's work
# # workon parlai (not working due to conda install)
source ~/.virtualenvs/parlai/bin/activate
cd ~/beto/ParlAI
python projects/image_chat/server_with_style_chinese_sub_blenderbot.py -mf models:image_chat/transresnet_multimodal/model --response_template_path projects/image_chat/ner/


# !!! start sim server for question selection
cd Xiaobei/REM_enhancement
source sim_py3/bin/activate
python web_serv_sim.py 9110



# TEST
# chat with Zenbo
#S1 Zenbo install btclient or companion app
#S2 pair Zenbo via system setting and app settings
#S3 start REM app for pad to test chinese chat

# or chat on website
#S1 visit http://0.0.0.0:8082/
#S2 image submit
#S3 Q&A submit, then get response


# other module test commands

python projects/image_chat/server_with_style_chinese.py -mf zoo:style_gen/prev_curr_classifier/model

python projects/image_chat/server_with_style_chinese.py -mf models:image_chat/transresnet_multimodal/model --response_template_path projects/image_chat/ner/

python projects/image_chat/server_with_style_chinese.py -mf zoo:style_gen/prev_curr_classifier/model --model projects.style_gen.classifier:ClassifierAgent --classes-from-file image_chat_personalities_file

python projects/image_chat/just_style.py -mf zoo:style_gen/prev_curr_classifier/model --model projects.style_gen.classifier:ClassifierAgent --classes-from-file image_chat_personalities_file

python projects/image_chat/just_style_2.py -mf zoo:style_gen/prev_curr_classifier/model --model projects.style_gen.classifier:ClassifierAgent --classes-from-file image_chat_personalities_file

python projects/image_chat/just_style_2.py -mf zoo:style_gen/prev_curr_classifier/model --model projects.style_gen.classifier:ClassifierAgent --classes-from-file image_chat_personalities_file --no_cuda

