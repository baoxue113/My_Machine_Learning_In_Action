# coding=utf-8
# 准备数据：切分文本
import bayes
import numpy

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
print mySent.split() # 切割数据但是'.'也算进去了

import re
regEx = re.compile('\\W*')
listofTokens = regEx.split(mySent) # 正则化后切词，去掉'.'
print listofTokens

print [tok for tok in listofTokens if len(tok) > 0] # 去掉空字符串
print [tok.lower() for tok in listofTokens if len(tok) > 0] # 字符串转小写

emailText = open('email/ham/6.txt').read()
listofTokens = regEx.split(emailText) # 正则化后切词，去掉'.'
print listofTokens


