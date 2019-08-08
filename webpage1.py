####################### DASH WEBPAGE SCRIPT #####################

# 1.0 REQUIRED LIBRARIES
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import cv2
import base64
import numpy as np
import time

# 1.1 BLURRING ALGORITHM SUPPORT
import sys
sys.path.insert(0, './utils/face_detection/anonai_skupina1/')
from FaceDetector import FaceDetector
from utils.paths.paths import model_path, mtcnn_path
import os
print("WORK DIR: ", os.getcwd())

# Initialize face detection model
faceDetector = FaceDetector(model_path=model_path, mtcnn_path=mtcnn_path, face_tracking="kalman", max_frame_padding=10)

# 1.2 LIVE STREAM SERVER
from flask import Flask, Response, render_template
server = Flask(__name__)

# Camera object:
# connects with camera via openCV to obtain a frame, 
# applies blurring algorithm and outputs processed frame,
# writes frame to chosen video path;
class VideoCamera(object):
    def __init__(self, face_detector, video_path="./assets/blurred_stream.mp4"):
        # @param face_detecor: FaceDetector object for face blurring
        self.detector = face_detector
        self.output_video_path = video_path
        self.video = cv2.VideoCapture(0)

        # Tests camera
        if self.video.isOpened():
            success, frame = self.video.read()
        else:
            print("Failed opening the camera feed")
            exit(-1)

        # Prepares and initiates video file
        video_xdim = frame.shape[0]
        video_ydim = frame.shape[1]
        print("camera resolution: ", video_xdim, video_ydim)

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.output_video = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (video_xdim, video_ydim))
        
    def clear(self):
        self.video.release()
        self.output_video.release()

    def get_frame(self):
        # Gets from camera
        success, frame = self.video.read()

        if success == True:
            # Blurs faces in the frame
            res, result_container = self.detector.face_blur(frame)

            # Writes to file
            if res == self.detector.RESULT_FRAME:
                blurred = result_container
                self.output_video.write(result_container)
            
            if res == self.detector.RESULT_QUEUE:
                for i in range(0, len(result_container)):
                    blurred = result_container.pop()
                    self.output_video.write(blurred)
        
        # Returns appropriately encoded frame
        ret, jpeg = cv2.imencode('.jpg', blurred)
        return jpeg.tobytes()

# Image generator:
# yields blurred frames from webcam;
def gen(camera):
    t_end = time.time()+20
    while time.time() < t_end:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    camera.clear()

# Live stream server:
# shows the output of the image generator;
@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(faceDetector)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 2 APP SETTINGS
app = dash.Dash(__name__, server=server)
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions']=True

logo_style = {
    'margin-left':'2.5%',
    'margin-right':'2.5%',
    'margin-top':'5px',
    'width':'200px'
}
colors = {
    'white': '#FFFFFF',
    'black': '#000000',
    'turquis': '#0a7e73',
    'green': '#85b889',
    'red': '#ff6666'
}

button_style = {
    'font-family': 'Trebuchet MS',
    'padding': '10px 50px 10px 50px',
    'font-size': '15px',
    'background-color': colors['green'],
    'box-shadow': '0 1px #ff6666'
}

upload_style={
            'width': '380px',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '20px',
            'textAlign': 'center',
            'backgroundColor': colors['white'],
            'margin-left':'50%',
            'color': 'black',
            'transform': 'translateX(-50%)'
        }

demo_style = {
    "display": "block",
  "margin-left": "auto",
  "margin-right": "auto",
  'width':'500px'
}


# 3 PAGE COMPONENTS 
##Used to increase simplicity of app.layout

# 3.1 Head

head = html.Header(id = "header", children = [
        html.A([html.Img(src="./assets/page/ektimo2.png", style = logo_style)], href='https://ektimo.ai/', target="_blank")
    ])
# 3.2 Begining banner

banner = html.Section(id = "banner", children=[
        html.Div(className="inner", children =[
            html.Header(children=[
                html.H1('Hide your face'),
                html.P('Use one of the two available algorithms to either blur or distort faces in your image/video files and protect the identity of their owners.'),
                html.P('Alternatively, use the real-time anonymization capture.')
                
                #html.Button('Begin', className="button big alt scrolly")
            ]),
            html.Img(src="./assets/page/arow4.png",style= {"position":"relative","bottom":"0px","height":"100px"}),
            html.P("Scroll down")
        ])
    ])


# 3.3 Inital submain
submain = [html.Div([
    html.Header(className="align-center",children=[
                    html.H2("Choose the type of face anonymization"),
                    html.P("Choose between either blured or distorted algorithm")
                ]),
                
                html.Div(id="stolpec_initial",className="flex flex-2",children=[
                    html.Div(id="blur_div",className="video col", n_clicks_timestamp= 0, n_clicks=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(id="bl.pic",src="./assets/page/blurred_demo_photo22.jpg")
                            ]),
                            html.H3("Blured algorithm",id="blur_id", className="caption")
                        ], style=None, id="blur_border") 
                        ]),
                    html.Div(id="di_div",className="video col", n_clicks_timestamp= 0, n_clicks=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(src="./assets/page/distorted.jpg")
                            ]),
                            html.H3("Distorted algorithm", id = "di_id",className="caption")
                        ], style=None, id="dist_border")     
                    ])
                ]),
                html.Div(id="gallery_or_cam", children = [])
                ],style={"width":"70%", "margin":"auto"})
                ]

# 3.4 Footer
footer = html.Footer(id="footer", children=[
        html.Div(className="inner", children=[
            html.Div(className="align-center", children = [
            html.Div(className="flex flex-3", children=[
                html.Div(className="col", children=[
                    html.A([
                        html.Img( src="./assets/page/mizs_slo2.jpg", style = logo_style)
                    ], href='http://www.mizs.gov.si/', target="_blank")
                ]),
                html.Div(className="col", children=[
                    html.A([
                        html.Img(src="./assets/page/srip_logo2.jpg", style = logo_style)
                    ], href='http://www.jpi-sklad.si/', target="_blank")
                ]),
                html.Div(className="col", children=[
                    html.A([
                        html.Img(src="./assets/page/logo_ekp2.jpg", style = logo_style)
                    ], href='http://ec.europa.eu/esf/home.jsp?langId=sl', target="_blank")
                ])
            ]),
            html.P(""),
            html.P(""),
            html.A([
                html.P("CSS template for this webpage was provided by TEMPLATED")
            ], href="https://templated.co/", target="_blank")
            ])
        ])

    ])


# 4 PAGE FRAMEWORK

#Includes visable components, which are presented to user and hiddn components that serve as support for callbacks

app.layout = html.Div(id="inital", children = [  
        html.Div(id="inner", children=[
            head,
            banner,
            html.Div(id="main", children=[
                html.Section(className="wrapper style1", children=[
                    html.Div(id="sub_main", className="inner", children=submain),
                    html.Div(className="inner", children=[
                        html.Header(className = "align-center", children =[
                            html.Button("ANONYMIZE", id = "back", disabled= True, className="button big alt scrolly", n_clicks=0)
                        ])
                    ])
                ])
            ]),
            footer #Visable components
        ]),
        dcc.RadioItems(id="alg-choice", style={'display': 'none'}, children = [
            {'label': ' Feature Distortion ', 'value': 'distorted_'},
            {'label': ' Blurring', 'value': 'blurred_'}
        ], value="distorted_"),
        dcc.RadioItems(id='from-choice', style={'display': 'none'}, options=[
            {'label': ' From your gallery ', 'value': 'fl'},
            {'label': ' Live from camera', 'value': 'lv'}
            ], value='fl'),
        dcc.Upload(id="up", contents=None, filename=None),
        html.Div(id= "back_button", n_clicks = 0,style={'display': 'none'}),
        html.Div(id= "switch",style={'display': 'none'}, n_clicks = 0),
        html.Button(id="blur_button", n_clicks=0, n_clicks_timestamp= 0)
]) #Hidden components

# 5 CALLBACKS

#Determines whether blurred or distorted algorithm is chosen.
@app.callback(Output("alg-choice", "value"),
              [Input("blur_div", "n_clicks_timestamp"),
               Input("di_div", "n_clicks_timestamp")])
def choice(blur, di):
    if int(blur) > int(di):
        return "blurred_"
    else:
        return "distorted_"

#Draws border around blurred algorithm button if chosen.
@app.callback(Output("blur_border", "style"),
              [Input("blur_div", "n_clicks_timestamp"),
               Input("di_div", "n_clicks_timestamp")])
def blur_border(blur, di):
    if int(blur) > int(di):
        return {"border-style":"solid","border-color":"rgb(129, 23, 23)", "border-width":"5px"}
    else:
        return None

#Draws border around distorted algorithm button if chosen.
@app.callback(Output("dist_border", "style"),
              [Input("blur_div", "n_clicks_timestamp"),
               Input("di_div", "n_clicks_timestamp")])
def dist_border(blur, di):
    if int(blur) < int(di):
        return {"border-style":"solid","border-color":"rgb(129, 23, 23)", "border-width":"5px"}
    else:
        return None

#Returns components where user can decide if he/she will use files from gallery or live cammera to either blur or distort them. 
@app.callback(Output("gallery_or_cam", "children"),
              [Input("blur_div", "n_clicks_timestamp"),
               Input("di_div", "n_clicks_timestamp")])
def add_submain(blur,di):
    if blur > di:
        return [html.Header(className="align-center",children=[
                    html.H2("Blurring"),
                    html.P("What do you want to blur?")
                ]),
                html.Div(id="stolpec_2",className="flex flex-2",children=[
                    html.Div(id="gallery_div",className="video col",n_clicks_timestamp=0, n_clicks=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(id="bl.pic",src="./assets/page/gallery3.jpg")
                            ]),
                            html.H3("Blur photos and videos from my gallery",id="blur_id", className="caption")
                            ], style=None, id="gal_border") 
                    ]),
                    html.Div(id="live_div",className="video col",n_clicks_timestamp=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(src="./assets/page/camera3.jpg")
                            ]),
                            html.H3("Blur videos from my web camera", id = "di_id",className="caption")
                            ], style=None, id="cam_border") 
                    ])
                ]),
                html.Header(id="upload_div", className="align-center", children=[
                    dcc.Upload(id='upload-image', contents=None,  children=[])
                ]),
                html.Header(html.Div(id="preview"))
        ]
    elif blur < di:
        return [html.Header(className="align-center",children=[
                    html.H2("Distortion"),
                    html.P("What do you want to distort?")
                ]),
                html.Div(id="stolpec_2",className="flex flex-2",children=[
                    html.Div(id="gallery_div",className="video col",n_clicks_timestamp=0, n_clicks=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(id="bl.pic",src="./assets/page/gallery3.jpg")
                            ]),
                            html.H3("Distort photos and videos from my gallery",id="blur_id", className="caption")
                        ], style=None, id="gal_border")
                    ]),
                    html.Div(id="live_div",className="video col",n_clicks_timestamp=0,children=[
                        html.Div([
                            html.Div(className="image fit", children=[
                                html.Img(src="./assets/page/camera3.jpg")
                            ]),
                            html.H3("Distort videos from my web camera", id = "di_id",className="caption")
                        ], style=None, id="cam_border")
                    ])
                ]),
                html.Header(id="upload_div", className="align-center", children=[
                    dcc.Upload(id='upload-image', contents=None, children=[])
                ]),
                html.Header(html.Div(id="preview"))
        ]
    else:
        return []


#Determines whether gallery or live camera option is chosen.
@app.callback(Output('from-choice', 'value'),
              [Input('gallery_div', 'n_clicks_timestamp'),
               Input('live_div', 'n_clicks_timestamp')])
def from_choice(gallery, live):
    if gallery >= live:
        return "fl"
    else:
        return "lv"

#Draws border around gallery button if chosen.
@app.callback(Output("gal_border", "style"),
              [Input("gallery_div", "n_clicks_timestamp"),
               Input("live_div", "n_clicks_timestamp")])
def gal_border(gal, live):
    if int(gal) > int(live):
        return {"border-style":"solid","border-color":"rgb(129, 23, 23)", "border-width":"5px"}
    else:
        return None

#Draws border around live cammera button if chosen.
@app.callback(Output("cam_border", "style"),
              [Input("gallery_div", "n_clicks_timestamp"),
               Input("live_div", "n_clicks_timestamp")])
def cam_border(gal, live):
    if int(gal) < int(live):
        return {"border-style":"solid","border-color":"rgb(129, 23, 23)", "border-width":"5px"}
    else:
        return None

#If gallery option is chosen, content of upload component is set. If live camera option is chosen, process description is presented.
@app.callback(Output('upload-image', 'children'),
              [Input('gallery_div', 'n_clicks_timestamp'),
              Input('live_div', 'n_clicks_timestamp')])
def upload_content(gallery,live):
    if gallery > live:
        return [
            html.Div(['Drag and Drop or ', html.A('Select Files', style={"color":"black"})], id='button1', style=upload_style)
        ]
    elif gallery < live:
        return [html.Header(className = "align-center", children = html.P("A 20 second clip will be captured, blured and displayed live."))]
    else:
        return []


#Construction of preview panel where user can see picture or video which was chosen for anonimization. Estimated time of anonimization is calculated and presented. 
@app.callback(Output('preview', 'children'),
              [Input('upload-image', 'contents'),
              Input("from-choice", "value"),
              Input("alg-choice", "value"),
              Input("upload-image", "filename")])
def upload_content2(contents, from_choice,algo, name):
    if contents is not None and from_choice == "fl":
        tmp = algo[:-1]
        if contents[5:10]=='image':
            x=len(contents)//(2*10**6)
            if x > 0:
                x = str(x) + ' min.'
            else:
                x=str((len(contents)%(2*10**6))//33333) + ' s.'
            return [html.Img(src=contents, style=demo_style),
            html.Header(className = "align-center", children = html.P('Anonymization of '+name+' using the '+tmp+' algorithm will take aproximately '+x))]    
        else:
            x=str(len(contents)//(8*10**5))+' min.'
            return [html.Header(className = "align-center", children = [html.Video(src=contents, style={'max-width':'600px','width':'80%'}, controls='controls'),
             html.P('Anonymization of '+name+' using the '+tmp+' algorithm will take aproximately '+x)])]
        
    else:
        return []


# 'up' is support component with values and properties of 'upload-image' component
# We set contents of 'up' component 
@app.callback(Output('up', 'contents'),
              [Input('upload-image', 'contents')])
def copy(cont):
    return cont

#We set filename of 'up' component
@app.callback(Output('up', 'filename'),
              [Input('upload-image', 'filename')])
def copy2(name):
    return name

#Contents of 'back' button is modified depending on times it was clicked
@app.callback(Output('back', 'children'),
              [Input('back', 'n_clicks')])
def button_text(n_clicks):
    if n_clicks % 2 == 1:
        return 'BACK'
    else:
        return 'ANONYMIZE'

#Switching between initial page where user decides how and what to anonimize and page where anonimized contents are presented.
@app.callback(Output('sub_main', 'children'),
              [Input('back', 'n_clicks')])
def main_content(n_clicks):
    if n_clicks % 2 == 1:
        return html.Div(id = "output")
    else:
        return submain

#'back' button is disabled until requirements are met. 
@app.callback(Output('back', 'disabled'),
              [Input('upload-image', 'contents'),
               Input('from-choice', 'value')])
def back_disable(contents,choice):
    if contents is not None or choice == "lv":
        return False
    else:
        return True

#Contents are processed, anonimized and returned to user. 
@app.callback(Output('output', 'children'),
              [Input('back', 'n_clicks')],
              [State('up', 'contents'),
              State('up', 'filename'),
              State('alg-choice','value'),
              State('from-choice','value')])
def show_output(n_clicks,contents,filename,algorithm,choice):
    if n_clicks % 2 == 1:
        if algorithm=='blurred_' and choice=='lv':
            return html.Header([html.H5('Blurring live',style={'color':colors['white'],'margin-top':'50px'}),
                            html.P(html.Img(src="/video_feed",style={'max-width':'600px'}), id='live')
            ],className = "align-center")
        elif algorithm == "blurred_" and choice == "fl":
            print(len(contents))
            # Image
            if contents[5:10]=='image':
                processed=process_img(from_b64(contents), algorithm)
                cv2.imwrite('./assets/'+algorithm+filename,processed)
                return [html.Header(className = "align-center", children=[
                    html.H5(algorithm+filename,style={'color':colors['white'],'margin-top':'50px'}),
                    html.P(html.Img(src='./assets/'+algorithm+filename, style={'max-width':'600px'})),
                    html.A(html.Button('DOWNLOAD', className="button big alt scrolly"), download=algorithm+filename, href='./assets/'+algorithm+filename)
                ])]
            # Video
            else:
                vid = contents.split(',')[1]
                with open("./assets/"+filename, 'wb') as wfile:
                    wfile.write(base64.b64decode(vid))
                process_vid(filename, algorithm)
                return [html.Header([
                    html.H5(algorithm+filename,style={'color':colors['white'],'margin-top':'50px'}),
                    html.P(html.Video(src='./assets/'+algorithm + filename, style={'max-width':'600px','width':'80%'}, controls='controls')),
                    html.A(html.Button('DOWNLOAD', className="button big alt scrolly"), download=algorithm+filename, href='./assets/'+algorithm + filename)
                ],className = "align-center")]
        else:
            return html.Header(className="align-center", children =[html.Div(html.H1('Distortion not yet available'))])
    else:
        return None


# 6 FUNCTIONS FOR CONTETNS PROCESSING

# 6.1 PROCESSING IMAGE FILES 

# Converts image from base64 to opencv frame format
def from_b64(url):
   image = url.split(',')[1]
   buffer = np.fromstring(base64.b64decode(image), np.uint8)
   frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
   return frame

# Accepts accepts image in opencv frame format + algorithm string, calls selected algorithm;
def process_img(contents, algorithm):
    if algorithm=='blurred_':
        contents=blur_img(contents)
    elif algorithm=='distorted_':
        contents=distort_img(contents)
    return contents

# Algorithm 1 (image): 
# accepts image in opencv frame format, returns blurred frame in same format;
def blur_img(contents):
    blurred = faceDetector.blur_single_image(contents)
    print('blurring')
    return blurred

# Algorithm 2 (image): 
# accepts image in opencv frame format,
# TO ADD: returns distorted frame in same format;
def distort_img(contents):
    print('distorting')
    return contents

# 4.3 PROCESSING VIDEO FILES 
# Accepts filename string + algorithm string, calls selected algorithm;
def process_vid(filename, algorithm):
    if algorithm=='blurred_':
        contents=blur_vid(filename)
    elif algorithm=='distorted_':
        contents=distort_vid(filename)

# Algorithm 1 (video): 
# accepts filename, writes blurred video to file;
def blur_vid(filename):
    # read from './assets/'+filename
    # output video file: "blurred_"+filename
    # save to './assets/'+"blurred_"+filename
    print('blurring')
    faceDetector.blur_video("./assets/" + filename)

# Algorithm 2 (video): 
# accepts filename,
# TO ADD: writes distorted video to file;
def distort_vid(filename):
    # read from './assets/'+filename
    # output video file: "distorted_"+filename
    # save to './assets/'+"distorted_"+filename
    print('distorting')
    pass



if __name__ == '__main__':
    app.run_server()

