#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:49:31 2023

@author: hyeontaemin
"""

#import taglist
from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import torch
import categoryModel



app = Flask(__name__)   # Flask객체 할당
 
CORS(app, resources={r'*': {'origins': '*'}}) # 모든 곳에서 호출하는 것을 허용







# tag 정보 가져오기
#tag = taglist.tagling

#tag = sorted(tag)



@app.route("/api/ai/category", methods=['POST','GET'])
def categoryClassification():
    
    params = request.get_json() # 전달된 json값을 저장
    title = params["id"]
    content = params["content"]
    
    resource = title + content
    
    result = categoryModel.predict(resource)
    
    return result
    



    
    
#app.run(host="0.0.0.0", port=2222) #서버 실행
app.run(port=8000, debug=True) #로컬 테스트 확인용 
