# gnome-terminal\
#     --tab\
#         --title="lampp" -- bash -c "cd /opt/lampp; sudo ./lampp start; $SHELL"
gnome-terminal\
    --tab\
        --title="event and place" -- bash -c "source /home/beto/.virtualenvs/remi2/bin/activate; cd /home/beto/predictors/iu; python server_predict_event_or_place.py 9100; $SHELL"

gnome-terminal\
    --tab\
        --title="relation" -- bash -c "source /home/beto/.virtualenvs/face_detector/bin/activate; cd /home/beto/predictors/iu; python server_predict_people.py 9200; $SHELL"

gnome-terminal\
    --tab\
        --title="blender_bot" -- bash -c "source /home/beto/Xiaobei/parlai_xb/bin/activate; cd /home/beto/Xiaobei/ParlAI; python projects/image_chat/server_updated_blenderbot.py -mf zoo:blenderbot2/blenderbot2_400M/model --knowledge-access-method memory_only --search-server None; $SHELL"

gnome-terminal\
    --tab\
        --title="question_sim" -- bash -c "cd /home/beto/Xiaobei/REM_enhancement; source sim_py3/bin/activate; python web_serv_sim.py 9110"
        
