from results_set import set_result, set_result2
from flask import Flask, render_template, url_for, request, redirect, session, flash

# python基本ライブラリ
import numpy as np

# ファイル作成
import shutil
import datetime
import cv2
import os

from dotenv import load_dotenv
from datetime import timedelta

# YOLO
from ultralytics import YOLO

# 環境変数読み込み
load_dotenv()

# アプリケーション起動
app = Flask(__name__)
SECRET_KEY = os.getenv("SECRET_KEY")
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(days=7)  # 1週間保存


# モデル初期化
dtect_model = YOLO('models/yolov8n.pt')
seg_model = YOLO('models/yolov8n-seg.pt')
cls_model = YOLO('models/yolov8n-cls.pt')

# home画面
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# YOLOv検証アプリ
@app.route('/', methods=['POST'])
def image_detect_post():

    # フォルダパスの作成
    img_dir = 'static/images/'
    result_dir = 'static/results/'
    dir_list = [img_dir, result_dir]
    
    # フォルダの初期化（削除）
    for item in dir_list:
        if os.path.exists(item):
            shutil.rmtree(item)
    
    # フォルダの作成
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 入力値受け取り
    input_model = request.form['model'] 
    input_image = request.files['image']
    
    # Flaskの設定
    input_flg = True

    # 入力チェック１ モデル
    if not (input_model == '1' or input_model == '2' or input_model == '3' ) :
        flash('モデルを選択してください')
        input_flg = False

    # 入力チェック２ 画像ファイル
    if not input_image:
        flash('画像を選択してください')
        input_flg = False
    
    # 初期画面に戻る
    if not input_flg:
        return redirect(url_for("home"))
    
    # 画像の変換
    stream = input_image.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    
    # 画像の保存
    input_filename =  input_image.filename
    ext = os.path.splitext(input_filename)

    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    img_path = img_dir + dt_now + ext[1]
    result_path = result_dir + 'predict/' + dt_now + ext[1]

    cv2.imwrite(img_path, img)
    
    title = None
    table_clm = None
    dic = {}

    # YOLOv8モデル検証
    if(input_model == '1'):
        title = '物体検知'
        table_clm = '検知数'

        # 物体検知
        dic = detect_image(img_path, result_dir)

    elif(input_model == '2'): 
        title = 'セグメンテーション'
        table_clm = '検知数'

        # セグメンテーション
        dic = seg_image(img_path, result_dir) 

    else:
        title = '画像分類'
        table_clm = '精度'

        # 画像分類
        dic = cls_image(img_path, result_dir)

        # 精度フォーマット文字列変換
        for key, val in dic.items():
            dic[key]  =  '{:.8f}'.format(val)

    # 画像出力
    return render_template('index.html',
                           exit=True,
                           title=title,
                           content=img_path,
                           content2=result_path,
                           table_clm=table_clm,
                           table_data=dic )

# 物体検知
def detect_image(img_path, result_dir):

    results = dtect_model(img_path, save=True, exist_ok=True, project=result_dir)

    return set_result(results)

# セグメンテーション
def seg_image(img_path, result_dir):
    
    results = seg_model(img_path, save=True, exist_ok=True, project=result_dir)

    return set_result(results)

# 画像分類
def cls_image(img_path, result_dir):
    
    results = cls_model(img_path, save=True, exist_ok=True, project=result_dir)

    return set_result2(results)


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)