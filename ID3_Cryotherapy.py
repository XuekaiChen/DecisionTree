#使用决策树的ID3算法对冷冻疗法是否有效进行分类
import math
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
        if int(cells[1]) < 33:
            cells[1] = '青年'
        elif (int(cells[1]) > 33) and (int(cells[1]) < 51):
            cells[1] = '中年'
        else:
            cells[1] = '老年'
        if float(cells[2]) < 6:
            cells[2] = '短'
        else:
            cells[2] = '长'
        if float(cells[3]) < 9:
            cells[3] = '少'
        else:
            cells[3] = '多'
        if cells[4] == '1':
            cells[4] = '常规'
        elif cells[4] == '2':
            cells[4] = '脚底'
        else:
            cells[4] = '二者'
        if int(cells[5]) < 377:
            cells[5] = '小'
        else:
            cells[5] = '大'
        if cells[6] == '1\n':
            cells[6] = '成'
        else:
            cells[6] = '败'
        DataSet.append(cells)
    return DataSet, FeatureName

#分割数据集，保留符合特征属性value属性值的行，同时删除该属性值的列
def splitDataSet(dataSet,value,axis): #(数据集，属性值，第i个属性)
    subDataSet = []
    for line in dataSet:
        if line[axis] == value:
            #删除这一列的值
            reduceFeatVec = line[:axis] #保留前面所有列的值
            reduceFeatVec.extend(line[axis+1:]) #保留后面所有列的值
            subDataSet.append(reduceFeatVec)
    return subDataSet #得到符合该特征的列，并删除掉某属性其属性值的dataset

#计算信息熵函数
def calEntropy(dataSet): #此dataset也可以是某属性值下的数据条目
    count = len(dataSet)  # 样本数
    label = {} #键值对{类别：该类别样本数}
    #计算子属性下每个类别个数
    for line in dataSet:
        if line[-1] not in label.keys():
            label[line[-1]] = 0
        label[line[-1]] += 1
    H_pre = 0.0
    for key in label.keys():
        prob = float(label[key])/count
        H_pre -= prob * math.log(prob, 2)
    return H_pre

#计算条件熵(某属性其某属性值的信息熵)
def conditionalEntropy(dataSet,i,feaValue):
    H_pos = 0.0
    for value in feaValue:
        subDataSet = splitDataSet(dataSet,value,i)
        prob = len(subDataSet) / float(len(dataSet))
        H_pos += prob * calEntropy(subDataSet)
    return H_pos

#计算某个属性的信息增益
def calInformationGain(dataSet,H_pre,i):
    feaList = [line[i] for line in dataSet] #第i个属性的一整列值
    feaValue = set(feaList) #取出这一属性的所有可取值
    H_pos = conditionalEntropy(dataSet,i,feaValue)
    InfoGain = H_pre - H_pos
    return InfoGain

#为决策树分支节点选择最优决策属性
def chooseBestFeature(dataSet,featureName):
    featureNum = len(featureName) #属性个数
    baseEntropy = calEntropy(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(featureNum): #对每个属性计算信息增益
        # infoUp = calInformationGain(dataSet,baseEntropy,i)  #ID3的信息增益算
        infoUp = calInformationGain(dataSet,baseEntropy,i)/calEntropy(dataSet)  #按C4.5的信息增益比算
        if infoUp > bestInfoGain:
            bestInfoGain = infoUp
            bestFeature = i
    return bestFeature #为int型，属性标号(0到5，共6个类别)

#无可选属性时判断样本所属类别(选出现过最多的)
def majorityCnt(classList):
    label = {}
    for x in classList:
        try:
            label[x] += 1
        except:
            label[x] = 1
    return max(label)

#创建决策树
def createTree(dataSet,featureName):
    classList = [line[-1] for line in dataSet] #将最后一列的类别整为一个列表
    # 停止条件1：
    if classList.count(classList[0]) == len(classList): #当只剩单一类别时，输出该类别
        return classList[0]
    # 停止条件2：
    if len(featureName) == 0: #没有属性可以再划分时，返回最多的类别
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataSet,featureName) #最佳属性的标号
    # 停止条件3：
    if bestFeature == -1:  #没有新的属性能让信息熵减小
        return majorityCnt(classList)
    bestFeatureName = featureName[bestFeature] #得到最佳属性的名称
    myTree = {bestFeatureName: {}} #map结构，且key为属性名称
    del (featureName[bestFeature]) #结点构造完毕，删除该属性
    bestFeaValue = attribute_dict[bestFeatureName] #找到需要分类的属性子集
    #建立每个属性值下的结点
    for value in bestFeaValue:
        subfeaName = featureName[:]
        if splitDataSet(dataSet,value,bestFeature): #若划分出的集合不是空集，则继续划分
            myTree[bestFeatureName][value] = createTree(splitDataSet(dataSet,value,bestFeature),subfeaName)
        else:
            # 停止条件4：
            return majorityCnt(classList) #当有一个子树为空集时，输出该属性中占多数的类别
    return myTree

# 对测试集数据用决策树划分类别
def classify(inputTree,featName,testVec):
    firstStr = list(inputTree.keys())[0]  #当前树的根节点的特征名称
    secondDict = inputTree[firstStr]  #根节点的子树
    featIndex = featName.index(firstStr)  #找到根节点特征对应的下标
    key = testVec[featIndex]  #找出待测数据的特征值
    valueOfFeat = secondDict[key]  #拿这个特征值在根节点的子节点中查找，看它是不是叶节点
    if isinstance(valueOfFeat, dict):  #如果不是叶节点
        classLabel = classify(valueOfFeat, featName, testVec)  #递归地进入下一层节点
    else:
        classLabel = valueOfFeat  #如果是叶节点：确定待测数据的分类
    return classLabel


DataSet, FeatureName = createDataSet()
attribute_dict = {'性别': {'男','女'}, '年龄': {'青年','中年','老年'},
                  '滋生时间': {'短','长'}, '个数': {'少','多'},
                  '种类':{'常规','脚底','二者'}, '面积':{'小','大'}}
# 构建 %80
test_Data = DataSet[0:-1:5] #从第0个到最后一个，间隔5个数据取一个作为测试集
del (DataSet[0:-1:5])
myTree = createTree(dataSet=DataSet, featureName=FeatureName)
print(myTree)

# 测试 %20
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