from flask import Flask,render_template,url_for,request
import base64
import cv2
import numpy as np
import tensorflow as tf
import torch
import pickle
import torchvision
from mlp import MLP
from torchinfo import summary

def crop_image(image):
    if image.sum()==0:
        return image
    x_max,y_max=image.shape
    x_right=x_max-1
    y_lower=y_max-1
    x_left=0
    y_upper=0
    while image[x_right,:].sum()==0:
        x_right-=1
    while image[x_left,:].sum()==0:
        x_left+=1
    while image[:,y_upper].sum()==0:
        y_upper+=1
    while image[:,y_lower].sum()==0:
        y_lower-=1
    return image[x_left:x_right,y_upper:y_lower]
def padding(image):
    len_x,len_y=image.shape
    raw_len=max(len_x,len_y)
    x_offset=(raw_len+150-len_x)//2
    y_offset=(raw_len+150-len_y)//2
    padded=np.zeros((raw_len+150,raw_len+150))
    padded[x_offset:x_offset+len_x,y_offset:y_offset+len_y]=image
    return padded

init_Base64 = 21;


app=Flask(__name__,template_folder='templates')
d={"1":'cnn1.h5',"2":"mlp.sav","3":"optimal_clf_dl.pkl"}

@app.route('/')
def home_endpoint():
    return render_template("draw.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=="POST":
        m=request.form.get("modeldropdown")
        draw=request.form['url']
        draw = draw[init_Base64:]
        draw_decoded=base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image=crop_image(image)
        image=padding(image)
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        cv2.imwrite('pred_im.png',resized)
        vect = resized.reshape(-1, 28, 28, 1)
        chosen_model=d[m]
        if chosen_model=='cnn1.h5':
            model=tf.keras.models.load_model(chosen_model)
            pred=model.predict(vect)
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
        elif chosen_model=='mlp.sav':
            mlp=torch.load(chosen_model)
            mlp.eval()
            vect.astype(np.float32)
            vect=torch.from_numpy(vect).float()
            pred=mlp.forward(vect).detach().numpy()
            stringlist=str(summary(mlp,(1,28,28),verbose=0)).split('\n')
        else:
            model=pickle.load(open('optimal_clf_dl.pkl','rb'))
            decomp=pickle.load(open('decomp-opt.pkl','rb'))
            pred=model.predict(decomp.transform(vect.reshape((1,-1))))
            stringlist=[]
        architecture=[]
        for line in stringlist:
            if line[:12]=='Total params':
                break
            if len(set(line))>1:
                architecture.append([line[:29].rstrip(),line[29:55].rstrip(),line[55:].rstrip()])
        index=np.argmax(pred)
        pred=(np.around(pred,3)*100).tolist()[0]
    return render_template('results.html',index=index,prediction=dict(enumerate(pred)),architecture=architecture)

@app.route('/models')
def models():
    return render_template('models.html')

if __name__=="__main__":
    app.run(host="192.168.1.99",port=5000)