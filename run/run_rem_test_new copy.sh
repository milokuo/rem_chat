# gnome-terminal\
#     --tab\
#         --title="lampp" -- bash -c "cd /opt/lampp; sudo ./lampp start; $SHELL"

# /home/beto/Xiaobei/clean_rem_process_list.sh

# gnome-terminal\
#     --tab\
#         --title="event and place" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/rem_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/iu; python server_predict_event_or_place.py 9201; $SHELL"

# gnome-terminal\
#     --tab\
#         --title="relation" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/rem_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/iu; python server_predict_people_copy.py 9202; $SHELL"

gnome-terminal\
    --tab\
        --title="clip_iu" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/clip_iu/clip_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/clip_iu; python clip_predictor.py 9205; $SHELL"

gnome-terminal\
    --tab\
        --title="detr_detector" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/clip_iu/clip_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/clip_iu; python detr_detector.py 9206; $SHELL"
        
gnome-terminal\
    --tab\
        --title="blip_caption" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/clip_iu/clip_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/clip_iu; python image_caption.py 9207; $SHELL"

gnome-terminal\
    --tab\
        --title="iu_redirect" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/rem_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/iu; python run_iu_server_handler.py 9209; $SHELL"

gnome-terminal\
    --tab\
        --title="chat_engine" -- bash -c "source /home/penguin37/companion/rem_chat/predictors/clip_iu/clip_iu/bin/activate; cd /home/penguin37/companion/rem_chat/predictors/clip_iu; python chat_engine.py; $SHELL"

# gnome-terminal\
#     --tab\
#         --title="blender_bot" -- bash -c "source /home/penguin37/companion/rem_chat/rem_env/bin/activate; cd /home/penguin37/companion/rem_chat/ParlAI; python projects/image_chat/server_updated_blenderbot.py -mf zoo:blenderbot2/blenderbot2_400M/model --knowledge-access-method classify --search-server 0.0.0.0; $SHELL"

# gnome-terminal\
#     --tab\
#         --title="social_rem" -- bash -c "source /home/penguin37/companion/rem_chat/rem_env/bin/activate; cd /home/penguin37/companion/rem_chat/ParlAI; python projects/image_chat/server_updated_xiaobei.py -mf zoo:blenderbot2/blenderbot2_400M/model --knowledge-access-method classify --search-server 0.0.0.0; $SHELL"

gnome-terminal\
    --tab\
        --title="social_rem" -- bash -c "source /home/penguin37/companion/rem_chat/rem_env/bin/activate; cd /home/penguin37/companion/rem_chat/ParlAI; python projects/image_chat/server_updated_zhengxuan.py; $SHELL"

gnome-terminal\
    --tab\
        --title="question_sim" -- bash -c "cd /home/penguin37/companion/rem_chat/VQA_enhancement; source sim_py3/bin/activate; python web_serv_sim.py 9110"
        
