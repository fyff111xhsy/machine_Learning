import math
from collections import Counter, defaultdict
import pandas as pd

# 计算信息熵
def entropy(labels):
    counter = Counter(labels)
    total = len(labels)
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

# 计算信息增益
def information_gain(data, labels, feature_index):
    base_entropy = entropy(labels)
    feature_values = defaultdict(list)

    for i, sample in enumerate(data):
        feature_values[sample[feature_index]].append(i)

    cond_entropy = 0.0
    for idxs in feature_values.values():
        subset_labels = [labels[i] for i in idxs]
        cond_entropy += len(idxs) / len(labels) * entropy(subset_labels)

    return base_entropy - cond_entropy

# 获取多数类标签
def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]

# 算法实现
def id3(data, labels, feature_names):
    # 情形1：纯结点
    if len(set(labels)) == 1:
        return labels[0]

    # 情形2：无特征
    if not feature_names:
        return majority_class(labels)

    # 选择最优特征
    gains = [
        information_gain(data, labels, i)
        for i in range(len(feature_names))
    ]
    best_idx = gains.index(max(gains))
    best_feature = feature_names[best_idx]

    tree = {best_feature: {}}

    # 按最优特征取值划分
    feature_values = set(sample[best_idx] for sample in data)
    for value in feature_values:
        sub_data = []
        sub_labels = []

        for i, sample in enumerate(data):
            if sample[best_idx] == value:
                sub_data.append(sample[:best_idx] + sample[best_idx+1:])
                sub_labels.append(labels[i])

        if not sub_data:
            tree[best_feature][value] = majority_class(labels)
        else:
            sub_features = feature_names[:best_idx] + feature_names[best_idx+1:]
            tree[best_feature][value] = id3(sub_data, sub_labels, sub_features)

    return tree

# 示例用法
# 构造一个简单的离散数据集（仿西瓜数据集）
data = {
    "色泽": ["青绿", "青绿", "乌黑", "乌黑", "浅白", "浅白", "青绿", "乌黑", "浅白", "青绿", 
            "乌黑", "浅白", "青绿", "乌黑", "乌黑", "浅白", "青绿", "青绿", "乌黑", "浅白",
            "青绿", "乌黑", "浅白", "乌黑", "青绿", "浅白", "乌黑", "青绿", "浅白", "青绿"],
    
    "根蒂": ["蜷缩", "蜷缩", "蜷缩", "稍蜷", "硬挺", "硬挺", "蜷缩", "稍蜷", "硬挺", "蜷缩",
            "稍蜷", "硬挺", "蜷缩", "稍蜷", "蜷缩", "硬挺", "蜷缩", "稍蜷", "硬挺", "蜷缩",
            "稍蜷", "硬挺", "蜷缩", "稍蜷", "蜷缩", "硬挺", "稍蜷", "蜷缩", "硬挺", "稍蜷"],
    
    "敲声": ["浊响", "沉闷", "浊响", "清脆", "清脆", "沉闷", "浊响", "清脆", "沉闷", "浊响",
            "清脆", "沉闷", "浊响", "清脆", "沉闷", "清脆", "浊响", "清脆", "沉闷", "浊响",
            "清脆", "沉闷", "清脆", "浊响", "沉闷", "清脆", "浊响", "沉闷", "清脆", "浊响"],
    
    "纹理": ["清晰", "清晰", "清晰", "清晰", "模糊", "模糊", "清晰", "稍糊", "模糊", "清晰",
            "稍糊", "模糊", "清晰", "稍糊", "清晰", "模糊", "清晰", "稍糊", "模糊", "清晰",
            "稍糊", "模糊", "稍糊", "清晰", "稍糊", "模糊", "清晰", "稍糊", "模糊", "清晰"],
    
    "脐部": ["凹陷", "凹陷", "凹陷", "平坦", "平坦", "平坦", "凹陷", "稍凹", "平坦", "凹陷",
            "稍凹", "平坦", "凹陷", "稍凹", "凹陷", "平坦", "凹陷", "稍凹", "平坦", "凹陷",
            "稍凹", "平坦", "稍凹", "凹陷", "稍凹", "平坦", "凹陷", "稍凹", "平坦", "凹陷"],
    
    "触感": ["硬滑", "软粘", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑", "软粘", "硬滑",
            "软粘", "硬滑", "硬滑", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "软粘", "硬滑",
            "硬滑", "软粘", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "软粘", "硬滑", "硬滑"],
    
    "密度": [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.666, 0.243, 0.245, 0.343,
            0.639, 0.657, 0.360, 0.593, 0.719, 0.481, 0.437, 0.666, 0.243, 0.245,
            0.343, 0.639, 0.657, 0.360, 0.593, 0.719, 0.481, 0.437, 0.666, 0.243],
    
    "含糖率": [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.091, 0.267, 0.057, 0.099,
              0.161, 0.198, 0.370, 0.042, 0.103, 0.153, 0.269, 0.091, 0.267, 0.057,
              0.099, 0.161, 0.198, 0.370, 0.042, 0.103, 0.153, 0.269, 0.091, 0.267],
    
    "好瓜": ["是", "是", "是", "否", "否", "否", "是", "否", "否", "是",
            "否", "否", "是", "否", "是", "否", "是", "是", "否", "是",
            "否", "否", "否", "是", "否", "否", "是", "是", "否", "是"]
}

df = pd.DataFrame(data)

feature_names = list(df.columns[:-1])
labels = df["好瓜"].tolist()

tree = id3(df[feature_names].values.tolist(), labels, feature_names)
print(tree)