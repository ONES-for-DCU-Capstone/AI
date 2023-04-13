#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:49:31 2023

@author: hyeontaemin
"""

import TagExtractor.taglist
from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import torch
import TextClassification.categoryModel
from konlpy.tag import Okt


app = Flask(__name__)   # Flask객체 할당
 
CORS(app, resources={r'*': {'origins': '*'}}) # 모든 곳에서 호출하는 것을 허용



def binary_search(a, x):
    start = 0
    end = len(a) - 1

    while start <= end:
        mid = (start + end) // 2
        if x == a[mid]:
            return mid
        elif x > a[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return -1



# tag 정보 가져오기
tag = TagExtractor.taglist.sortedTag




@app.route("/api/ai/category", methods=['POST','GET'])
def categoryClassification():
    
    params = request.get_json() # 전달된 json값을 저장
    title = params["title"]
    content = params["content"]
    
    resource = title + content
    
    result = TextClassification.categoryModel.predict(resource)
    
    return result
    


@app.route("/api/ai/hashtag", methods=['POST','GET'])
def createHashtag():
    
    params = request.get_json()
    title = params["title"]
    content = params["content"]
    
    resource = title + content
    okt = Okt()
    splitResource = okt.nouns(resource)
  
    
    hashtag = []
    
    
    for i in splitResource:
        j = binary_search(tag, i)
        if j != -1:
            i = "#"+i
            hashtag.append(i)
   
    
    hashtagList = {"hashtag" : hashtag }
    print(hashtagList)
   
    return jsonify(hashtagList)
    
            
    
    
    


    
    
#app.run(host="0.0.0.0", port=2222) #서버 실행
app.run(port=8000, debug=True) #로컬 테스트 확인용 
