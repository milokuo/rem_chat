# -*- coding:utf-8 -*-
import requests
import time
import json
import argparse
import datetime
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import PIL.Image as Image
from base64 import b64decode
import io
import os
import random
import re

HOST_NAME = "0.0.0.0"
PORT = 9209
STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark">
                <div id="parent" class="hero-body">
                    <article class="media" id="photo-info">
                      <figure class="media-left">
                        <span class="icon is-large">
                          <i class="fas fa-robot fas fa-2x"></i>
                        </span>
                      </figure>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <img id="preview" src="Examples.png"/ style="max-height:300px">
                          </p>
                        </div>
                      </div>
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Model</strong>
                            <br>
                            Enter a message, and the model will respond interactively.
                          </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column">
                  <form id = "interact">
                      <div class="field is-grouped">
                      <p class="control">
                        Type the model name: clip/detr/caption:
                        <input class="input" form="interact" type="text" id="userIn" placeholder="Type in a model" size="10">
                      </p>
                      <p class="control">
                        Type the image filename:
                        <input class="input" form="interact" type="text" id="userIn2" placeholder="Type in a url" size="10">
                      </p>
                        <p class="control is-expanded">
                          Upload an image:
                          <input class="input" type="file" id="userInImage" accept="image/*">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Submit
                          </button>
                        </p>
                      </div>
                  </form>
                  <p class="control">
                    <button id="newImage" class="button has-text-white-ter has-background-grey-dark">
                      New Image
                    </button>
                  </p>
                </div>
              </section>
            </div>
        </div>

        <script>
            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"

                var figure = document.createElement("figure");
                figure.className = "media-left";

                var span = document.createElement("span");
                span.className = "icon is-large";

                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : "");

                var media = document.createElement("div");
                media.className = "media-content";

                var content = document.createElement("div");
                content.className = "content";

                var para = document.createElement("p");
                var paraText = document.createTextNode(text);

                var strong = document.createElement("strong");
                strong.innerHTML = agent;
                var br = document.createElement("br");

                para.appendChild(strong);
                para.appendChild(br);
                para.appendChild(paraText);
                content.appendChild(para);
                media.appendChild(content);

                span.appendChild(icon);
                figure.appendChild(span);

                media.id = "model-response1";
                figure.id = "model-response2";

                article.appendChild(figure);
                article.appendChild(media);


                return article;
            }}
            function fetchResult(image_data, cate, img_id) {{
                var formData = new FormData();
                formData.append('image', image_data);
                formData.append('cate', cate);
                formData.append('img_id', img_id);
                fetch('/interact', {{

                    method: 'POST',
                    body: formData
                }}).then(response=>response.json()).then(data=>{{
                    var parDiv = document.getElementById("parent");
                    if (cate !== "") {{
                        parDiv.append(createChatRow("IU Model", cate));
                    }}
                    if (img_id !== "") {{
                        parDiv.append(createChatRow("Image URL", img_id));
                    }}
                    // Change info for Model response
                    parDiv.append(createChatRow("Metadata", data.text));
                    document.getElementById("userInImage").value = "";
                    window.scrollTo(0,document.body.scrollHeight);
                }});

            }}
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()
                var cate = document.getElementById("userIn").value;
                document.getElementById("userIn").value = "";
                var img_id = document.getElementById("userIn2").value;
                document.getElementById("userIn2").value = "";

                var img_input = document.getElementById("userInImage");
                var preview = document.getElementById("preview");
                if (img_input.files && img_input.files[0]) {{
                  var reader = new FileReader();
                  reader.onload = function (e) {{
                    preview.setAttribute('src', e.target.result);
                    fetchResult(e.target.result, cate, img_id);
                  }};
                  reader.readAsDataURL(img_input.files[0]);
                }} else {{
                  fetchResult('', cate, img_id);
                }}
            }});
            document.getElementById("newImage").addEventListener("click", function(event){{
                event.preventDefault()
                var oldResponse = document.getElementById("model-response1");
                while (oldResponse) {{
                    oldResponse.parentNode.remove(oldResponse);
                    oldResponse = document.getElementById("model-response1");
                }}
                var oldResponse = document.getElementById("model-response2");
                while (oldResponse) {{
                    oldResponse.parentNode.remove(oldResponse);
                    oldResponse = document.getElementById("model-response2");
                }}
                var preview = document.getElementById("preview");
                preview.setAttribute('src', '');
            }});
        </script>

    </body>
</html>
"""  # noqa: E501

def send_post_message_to_sim(msg):
    url = 'http://127.0.0.1:9110/'
    headers = {"Content-Type": "application/json"}
    post_msg = msg
    post_msg["post_time"] = str(time.time())
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))
    req = requests.post(url, data=_post_data, headers=headers)
    if req.status_code == requests.codes.ok:
        print('Sending ok')
        result = req.json()
        print("get response: {}".format(result))
        return result
    else:
        print('Sending fail')

def send_post_message(msg_, url):
    post_msg = msg_
    post_msg["post_time"] = str(time.time())  # 定義發送信息

    headers = {"Content-Type": "application/json"}  # 定義請求頭
    _post_data = json.dumps(post_msg, sort_keys=True, separators=(',', ':'))  # 將信息打包為json格式
    req = requests.post(url, data=_post_data, headers=headers)  # 使用requests，以POST形式發送信息
    if req.status_code == requests.codes.ok:  # 如果請求正常並收到回復
        # print('Sending ok')
        res = req.json()  # 讀取回復
        # print(json.dumps(res))
        return res
    else:
        print('Sending fail')
        return {}

class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        """
        Handle headers.
        """
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
    
    def process_post(self, form):
        img_id = cate = image_name = image_interactive = metadata = ""
        for field in form.keys():
            if field == "img_id":
                img_id = form[field].value
            elif field == "cate":
                cate = form[field].value
            elif field == "image_name":
                image_name = form[field].value
            elif field == "image":
                image_interactive = form[field].value
            elif field == "metadata":
                metadata = form[field].value
                print(">>> get metadata: {}".format(metadata))
        return {"img_id" : img_id, "cate" : cate, "image_name" : image_name, "image_interactive" : image_interactive, "metadata" : metadata}

    def do_POST(self):
        """
        Handle POST.
        """
        if self.path != "/interact":
            return self.respond({"status": 500})

        # Parse parameters from HTTP Server
        form = cgi.FieldStorage(
            fp=self.rfile, 
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        postvars = self.process_post(form)


        if postvars["image_name"]:
            #Extract the image features from an image stored in the computer.
            SHARED["image_name"] = postvars["image_name"]
            image_location = os.path.join(SERVER_IMAGE_LOCATION, SHARED["image_name"])

            if SERVER_IMAGE_LOCATION.find('http') == -1:
                image = Image.open(image_location).convert("RGB")
            else:
                image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

            reset_system_status = True
            model_response = {}
        
        # Used for the interactive mode
        elif postvars["image_interactive"] != "":
            # Extract the image features from an image provided via the interactive website.
            img_data = str(postvars["image_interactive"])
            _, encoded = img_data.split(",", 1)
            image = Image.open(io.BytesIO(b64decode(encoded))).convert("RGB")
            
            reset_system_status = True
            model_response = {"text": "IMAGE RECEIVED, you can start chatting..."}


        elif postvars['img_id']:
            # pending_list = '/home/penguin37/companion/rem_chat/predictors/pending_list.txt'
            
            # if not os.path.exists(process_list):
            #     with open(process_list, 'w') as f:
            #         f.write(postvars['img_id'] + ';')
            
            # else:
            #     with open(process_list, 'r') as record:
            #         for line in record.readlines():
            #             tmp = line.split(';')
            #             if postvars['img_id'] in tmp:
            #                 if tmp[1] != "":
            #                     # read saved metadata
            #                     data = json.loads(tmp[1])
            #                     # if postvars['cate'] in data:
            #                         # split_data = {postvars['cate']: data[postvars['cate']]}
            #                     print()
            #                     print("{} (get saved result)".format(postvars['cate']))
            #                     print()
            #                     print(json.dumps(data))
            #                     print()
            #                     return json.dumps(data)
            #                 else:
            #                     # wait for previous process to finish
            #                     data = {"status": "running"}
            #                     return json.dumps(data)
            #     # no record found
            #     with open(process_list, 'a') as f:
            #         f.write(postvars['img_id'] + ';')

            start = time.time()

            if postvars['cate']:
                msg = {'img_id': postvars['img_id'], 'cate': postvars['cate']}

                if postvars['cate'] in ['event', 'place']:
                    return_json = send_post_message(msg, 'http://127.0.0.1:9201/')
                elif postvars['cate'] == 'people':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9202/')
                elif postvars['cate'] == 'clip':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9205/')
                    # return_json = send_post_message(msg, 'http://192.168.1.105:9205/')
                elif postvars['cate'] == 'detr':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9206/')
                    # sim_msg = {}
                    # sim_msg['metadata'] = return_json
                    # send_post_message_to_sim(return_json)
                elif postvars['cate'] == 'caption':
                    return_json = send_post_message(msg, 'http://127.0.0.1:9207/')
                    # sim_msg = {}
                    # sim_msg['metadata'] = return_json
                    # send_post_message_to_sim(return_json)
                else:
                    print('wrong cate type!')
                    # return {}
                    return_json = {}

            print()
            print("predict cost time: {}".format(time.time()-start))
            print()
            
            print()
            print(postvars['cate'])
            print(return_json)
            # print(json.dumps(return_json))
            print()
            
            model_response = {"text": json.dumps(return_json)}

            # with open(process_list, 'a') as f:
            #     f.write(json.dumps(return_json) + '\n')

            # return_data = json.dumps(return_json)
            # return_data = json.dumps(return_json, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            # return return_data
        else:
            # return {}
            return_json = {}
            model_response = {"text": ""}
        
        

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        json_str = json.dumps(model_response)
        self.wfile.write(bytes(json_str, "utf-8"))
    
    def do_GET(self):
        """
        Handle GET.
        """
        paths = {
            "/": {"status": 200},
            "/favicon.ico": {"status": 202},  # Need for chrome
        }
        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({"status": 500})

    def handle_http(self, status_code, path, text=None):
        """
        Generate HTTP.
        """
        self.send_response(status_code)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
        return bytes(content, "UTF-8")

    def respond(self, opts):
        """
        Respond.
        """
        response = self.handle_http(opts["status"], self.path)
        self.wfile.write(response)


if __name__ == '__main__':
    now = datetime.datetime.now()
    print('Initial in {}'.format(now.strftime("%Y-%m-%d %H:%M:%S")))
    
    print("Initialized IU Redirector \
    \n -> 9201: event and place model \
    \n -> 9202: people model \
    \n -> 9205: clip model \
    \n -> 9206: detr model \
    \n -> 9207: caption model \
    \n")

    server_class = HTTPServer
    Handler = MyHandler
    Handler.protocol_version = "HTTP/1.0"
    httpd = server_class((HOST_NAME, PORT), Handler)
    print("\nVisit http://{}:{}/ to chat with the model!".format(HOST_NAME, PORT))
    print()
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()



