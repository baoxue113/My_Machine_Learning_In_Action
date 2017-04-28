# coding=utf-8
# 使用朴素贝叶斯分类器从个人广告中获取区域倾向 => 测试feedparser(结果失败)
import bayes
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
print(ny['entries'])
print(len(ny['entries']))



