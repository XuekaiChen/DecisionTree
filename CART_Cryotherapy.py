import itertools
import copy
import re
def createDataSet():
    FeatureName = ['性别', '年龄', '滋生时间', '个数', '种类', '面积']
    path = 'data\\Cryotherapy.txt'
    f = open(path, 'r')
    DataSet = []
    for cells in f.readlines():
        cells = cells.split('\t')
        if cells[0] == '1':
            cells[0] = '男'
        else:
            cells[0] = '女'
        if cells[4] == '1':
            cells[4] = '常规'
        elif cells[4] == '2':
            cells[4] = '脚底'
        else:
            cells[4] = '二者'
        if cells[6] == '1\n':
            cells[6] = '成'
        else:
            cells[6] = '败'
        DataSet.append(cells)
    return DataSet, FeatureName

#计算集合的非空子集，便于将属性值二分类(待优化：会有顺序不同的重复集合，不知道怎么去重)
def subSet(items):
    a = []
    b = []
    for i in range(1, len(items)):
        a.extend(list(map(list, itertools.combinations(items, i))))
    for j in range(len(a) - 1):
        if list(set(items)-set(a[j])) in a:
            a[j] = 0
    for k in a:
        if k != 0:
            b.append(k)
    return b

#当属性被用作划分属性后删除该属性及其列
def splitDataSet(dataSet,value,axis): #(数据集，属性值，第i个属性)
    subDataSet = []
    for line in dataSet:
        if line[axis] == value:
            #删除这一列的值
            reduceFeatVec = line[:axis] #保留前面所有列的值
            reduceFeatVec.extend(line[axis+1:]) #保留后面所有列的值
            subDataSet.append(reduceFeatVec)
    return subDataSet #得到符合该特征的列，并删除掉某属性其属性值的dataset

def calGini(dataSet,axis):
    a,b,a1,a2,b1,b2=0,0,0,0,0,0
    for line in dataSet:
        if line[axis] == 0:
            if line[-1] == '成':
                a1 += 1
            elif line[-1] == '败':
                a2 += 1
            else: continue
        elif line[axis] == 1:
            if line[-1] == '成':
                b1 += 1
            elif line[-1] == '败':
                b2 += 1
            else: continue
        else: continue
    a = a1 + a2
    b = b1 + b2
    Gini_a = 1 - pow(a1/a, 2) - pow(a2/a, 2)
    Gini_b = 1 - pow(b1/b, 2) - pow(b2/b, 2)
    Gini = a/(a+b) * Gini_a + b/(a+b) * Gini_b
    return Gini

def chooseBest(dataSet,featureName):
    featureNum = len(featureName)  # 属性个数
    bestFeature, l, r, ldata, rdata = 0, "", "", [], []
    Gini = 1
    for axis in range(featureNum): #遍历每个属性
        values = sorted(list(set([line[axis] for line in dataSet]))) #无重复列出所有属性值并排序
        try: #若属性值为连续型
            for i in range(len(values)-1): #对两两属性值计算均值，将所有属性值依照这个数划分成两个集合
                tempDataSet = copy.deepcopy(dataSet) #由于要将数据中的属性值按0/1划分，因此建立临时数据集变量
                mid = (float(values[i])+float(values[i+1]))/2 #计算两两属性值间的均值
                for line in tempDataSet:
                    if float(line[axis]) < mid: #小于此均值的所有值划分为一个集合，属性值记为0
                        line[axis] = 0
                    else: #大于此均值的所有值划分为一个集合，属性值记为1
                        line[axis] = 1
                if calGini(tempDataSet,axis) < Gini:
                    Gini = calGini(tempDataSet,axis) #得到当前最小Gini值
                    bestFeature = axis #得到最优分裂属性序号
                    l = '<'+str(mid)
                    r = '>'+str(mid)
                    ldata = splitDataSet(tempDataSet, 0, axis)
                    rdata = splitDataSet(tempDataSet, 1, axis)
                elif calGini(tempDataSet,axis) >= Gini:
                    continue
        except: #若属性值为离散型
            for sub1 in subSet(values): #给集合随机分裂成两个子集
                tempDataSet = copy.deepcopy(dataSet)
                for line in tempDataSet: #将属性列的属性值调整成可以计算Gini的0和1
                    if line[axis] in sub1: #若属于第一个子集则将属性值变为0
                        line[axis] = 0
                    else: #若属于第二个子集则将属性值变为1
                        line[axis] = 1
                if calGini(tempDataSet,axis) < Gini:
                    Gini = calGini(tempDataSet,axis) #得到当前最小Gini值
                    bestFeature = axis #得到最优分裂属性序号
                    l = ",".join(sub1)  #左子树属性值
                    r = ",".join(list(set(values)-set(sub1))) #右子树属性值
                    ldata = splitDataSet(tempDataSet, 0, axis)
                    rdata = splitDataSet(tempDataSet, 1, axis)
                elif calGini(tempDataSet,axis) >= Gini:
                    continue
    return bestFeature, l, r, ldata, rdata

def majorityCnt(classList):
    label = {}
    for x in classList:
        try:
            label[x] += 1
        except:
            label[x] = 1
    if label == {}:
        return {}
    else:
        return max(label)

def createTree(dataSet,featureName):
    classList = [line[-1] for line in dataSet]  # 将最后一列的类别整为一个列表
    # 停止条件1：
    if classList.count(classList[0]) == len(classList): #当只剩单一类别时，输出该类别
        return classList[0]
    # 停止条件2：
    if len(featureName) == 1: #当没有属性可以再划分时，返回最多的类别
        return majorityCnt(classList)
    bestFeature, l, r, ldata, rdata = chooseBest(dataSet, featureName)[0],\
                                      chooseBest(dataSet, featureName)[1],\
                                      chooseBest(dataSet, featureName)[2],\
                                      chooseBest(dataSet, featureName)[3],\
                                      chooseBest(dataSet, featureName)[4]  # 得到最佳分裂属性标号
    # 停止条件3：
    if ldata == [] or rdata == []: #当有一个子树为空集时，输出该属性中占多数的类别
        return majorityCnt(classList)
    bestFeatureName = featureName[bestFeature]  # 得到最佳属性的名称
    myTree = {bestFeatureName: {}}  # 建立子树
    del (featureName[bestFeature])
    subfeaName = featureName[:]
    myTree[bestFeatureName][l] = createTree(dataSet=ldata, featureName=subfeaName)
    myTree[bestFeatureName][r] = createTree(dataSet=rdata, featureName=subfeaName)
    return myTree

# 对测试集数据用决策树划分类别
def classify(inputTree,featName,testVec):
    firstStr = list(inputTree.keys())[0]  #当前树的根节点的特征名称
    secondDict = inputTree[firstStr]  #根节点的子树
    featIndex = featName.index(firstStr)  #找到根节点特征对应的下标
    key = testVec[featIndex]  #找出待测数据的特征值
    valueOfFeat = ''
    try:
        flag = 0
        for i in secondDict.keys():
            aa = float(re.findall(r'\d+.\d+', i)[0]) #提取map中分支的离散数值,若无则转到except
            if (flag == 0 and float(key) < aa) or flag == 1:  #小于分支数值则在第1个i进入下一层，大于则在第2个i进入下一层
                valueOfFeat = secondDict[i]
                break
            else: #若大于，则进入下一个循环的if
                flag += 1
    except:
        for i in secondDict.keys():
            if i.find(key) != -1: #若匹配到字符串，则进入
                valueOfFeat = secondDict[i]
            else:
                continue
    if isinstance(valueOfFeat, dict):  #如果不是叶节点
        classLabel = classify(valueOfFeat, featName, testVec)  #递归地进入下一层节点
    else:
        classLabel = valueOfFeat  #如果是叶节点：确定待测数据的分类
    return classLabel


DataSet, FeatureName = createDataSet()

# 构建 %80
test_Data = DataSet[0:-1:5] #从第0个到最后一个，间隔5个数据取一个作为测试集
del (DataSet[0:-1:5])
myTree = createTree(dataSet=DataSet, featureName=FeatureName)
print(myTree)

# # 测试 %20
FeatureName = ['性别', '年龄', '滋生时间', '个数', '种类', '面积']
correct = 0
prd_correct = 0
recall = 0
act_recall = 0
for line in test_Data:
    if line[-1] == '成':
        if classify(myTree,FeatureName,line) == '成':
            recall += 1
        act_recall += 1
    if classify(myTree,FeatureName,line) == '成':
        if line[-1] == '成':
            correct += 1
        prd_correct += 1
p = correct/prd_correct
q = recall/act_recall
print(correct,prd_correct,recall,act_recall)
print("准确率为：" + str(p))
print("召回率为：" + str(q))
print("调和平均：" + str(p*q*2/(p+q)))


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

print(getNumLeafs(myTree))