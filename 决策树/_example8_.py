# coding=utf-8
# 使用决策树预测隐形眼镜类型
import trees
import treePlotter
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# prescript:规格 astigmatic：闪光
lensesLabels = ['age','prescript','astigmatic','tearRate'] #特征的含义
lensesTree = trees.createTree(lenses,lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)

