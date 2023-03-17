#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:09:09 2023

@author: hyeontaemin
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

url ='해당주소'

driver = webdriver.Chrome('./chromedriver.exe')
driver.get(url)
driver.implicitly_wait(3)

before_h = driver.execute_script("return window.scrollY") 
'y좌표 저장' 


for i in range(25) :

    while True:
    
        driver.find_element(By.CSS_SELECTOR,"body").send_keys(Keys.END)
				'마지막 y좌표 위치 설정'
        
        time.sleep(1)
        
        after_h = driver.execute_script("return window.scrollY")
				'마지막 y좌표 반환'        

        if after_h == before_h:
            break
        before_h = after_h

    items = driver.find_elements(By.CSS_SELECTOR,".basicList_title__VfX3c")

    for item in items:
        name = item.find_element(By.CSS_SELECTOR,".basicList_link__JLQJf").text
        print(name)
    
    url = 'https://search.shopping.naver.com/search/all?origQuery=%EA%B3%84%EB%9E%80&pagingIndex='+str(i+1)+'&pagingSize=40&productSet=total&query=%EA%B3%84%EB%9E%80&sort=rel&timestamp=&viewType=list'
    driver.get(url)