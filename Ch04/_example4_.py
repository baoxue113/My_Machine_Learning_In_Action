# coding=utf-8
# 准备数据：切分文本
import bayes
import numpy

mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
print mySent.split() # 切割数据但是'.'也算进去了

import re
regEx = re.compile('\\W*')
listofTokens = regEx.split(mySent) # 正则化后切割数据，去掉'.'
print listofTokens


