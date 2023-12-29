#Library yang diambil dari pustaka Python
import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} #hanya file ekstensi png jpg jpeg yang diizinkan
app = Flask(__name__) #inisialisasi aplikasi flask
UPLOAD_FOLDER = 'static/uploads/' #direktori tempat file-file yang diunggah akan disimpan
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 #mengatur batas maksimum ukuran file yang diunggah

#model yang digunakan
model_used = './model/model_rps.h5'

# Menampilkan hasil prediksi model ke halaman HTML 
def predict_result(model, run_time, probs, img):
    class_list = {'Paper': 0, 'Rock': 1, 'Scissors:' :2}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result.html', labels=labels,
                            probs=probs, model=model, pred=idx_pred,
                            run_time=run_time, img=img)
# data yang diperlukan untuk merender halaman tersebut diambil dari parameter yang diteruskan ke fungsi.

@app.after_request
#fungsi ini memastikan bahwa halaman web yang dihasilkan oleh server tidak disimpan di cache browser, sehingga konten selalu diperbarui setiap halaman diakses
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

 #menampilkan halaman utama dari web saat user mengakses rute yang sesuai.
@app.route("/")
def index():
    return render_template('/index.html', )


@app.route('/predict', methods=['POST']) #pengiriman data dari pengguna yang memerlukan pemrosesan dari server
def predict():
    #
    model = load_model(model_used)
    #
    file = request.files["file"] #mengambil file dari form
    file.save(os.path.join('static', 'temp.jpg'))#disimpan jadi temporary file
    img = cv2.cvtColor(np.array(Image.open('./static/temp.jpg')), cv2.COLOR_BGR2RGB)#convert warna dari BGR ke rgb
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img.astype('float32') / 255, axis=0)#rescale 1-255 jadi 0-1
    start = time.time()
    pred = model.predict(img)[0]
    labels = (pred > 0.5).astype(int)
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result("MODEL PREDIKSI", runtimes, respon_model, 'temp.jpg')

if __name__ == "__main__":
        app.run(debug=True, host='0.0.0.0', port=2000)