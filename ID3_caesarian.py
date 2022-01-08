#使用决策树的ID3算法对是否剖腹产进行分类
import math
'''
程序中的标识：
feature指属性
value指属性值
i标识属性的标号
'''
def createDataSet():
    FeatureName = ['年龄', '孩子', '怀孕', '血压', '心脏病']
    #预处理数据集，化为列表
    path = 'data\\caesarian.txt'
    f = open(path, 'r')
    DataSet = []
    for line in f.readlines():
        line = line.split(',')
        if int(line[0]) < 25:
            line[0] = '年轻'
        elif (int(line[0]) > 25) and (int(line[0]) < 34):
            line[0] = '适龄'
        else:
            line[0] = '大龄'
        if line[2] == '0':
            line[2] = '及时'
        elif line[2] == '1':
            line[2] = '早产'
        elif line[2] == '2':
            line[2] = '晚产'
        if line[3] == '0':
            line[3] = '低'
        elif line[3] == '1':
            line[3] = '正常'
        elif line[3] == '2':
            line[3] = '高'
        if line[4] == '0':
            line[4] = '有'
        elif line[4] == '1':
            line[4] = '无'
        if line[5] == '0\n':
            line[5] = '否'
        elif line[5] == '1\n':
            line[5] = '是'
        DataSet.append(line)
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
def conditionalEntropy(dataSet,i,feaValue): #（剩余数据集，第i个属性，属性可取的全部属性值）
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
        infoUp = calInformationGain(dataSet, baseEntropy, i)  # ID3的信息增益算
        # infoUp = calInformationGain(dataSet,baseEntropy,i)/calEntropy(dataSet)  #按C4.5的信息增益比算
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
    if len(featureName) == 0: #当没有属性可以再划分时，返回最多的类别
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
        if splitDataSet(dataSet,value,bestFeature):  #若划分出的集合不是空集，则继续划分
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


# 构建80%
DataSet, FeatureName = createDataSet()
attribute_dict = {'年龄': {'年轻','适龄','大龄'}, '孩子': {'1','2','3','4'},
                  '怀孕': {'及时','早产','晚产'}, '血压': {'低','正常','高'},
                  '心脏病': {'有','无'}}
test_Data = DataSet[0:-1:5] #从第0个到最后一个，间隔5个数据取一个作为测试集，即20%
del (DataSet[0:-1:5])
myTree = createTree(dataSet=DataSet, featureName=FeatureName)
print(myTree)

# 测试20%
FeatureName = ['年龄', '孩子', '怀孕', '血压', '心脏病']
correct = 0
prd_correct = 0
recall = 0
act_recall = 0
for line in test_Data:
    if line[-1] == '是':
        if classify(myTree,FeatureName,line) == '是':
            recall += 1
        act_recall += 1
    if classify(myTree,FeatureName,line) == '是':
        if line[-1] == '是':
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
