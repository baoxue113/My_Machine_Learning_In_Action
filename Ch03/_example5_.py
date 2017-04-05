# coding=utf-8
# 图形化决策树
import treePlotter
myTree = treePlotter.retrieveTree(0)
# treePlotter.createPlot(myTree)
myTree['no surfacing'][3] = 'mybe'
print myTree
print
treePlotter.createPlot(myTree)

