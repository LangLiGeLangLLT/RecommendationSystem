import os
import numpy as np
import pandas as pd
import random
import math
from operator import itemgetter

class ItemBasedCF:

    # 将数据集随机分成训练集和测试集
    def SplitData(data, M, k, seed):
        test = dict()
        train = dict()
        random.seed(seed)
        for i in range(len(data)):
            user = data['userId'][i]
            item = data['movieId'][i]
            rating = data['rating'][i]
            if random.randint(0, M) == k:
                test.setdefault(user, dict())
                test[user][item] = rating
            else:
                train.setdefault(user, dict())
                train[user][item] = rating
        return train, test
    
    # 计算物品相似度 ItemCF
    def ItemSimilarityItemCF(train):
        # calculate co-rated users between items
        C = dict()
        N = dict()
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    C.setdefault(i, dict())
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        
        # calculate finial similarity matrix W
        W = dict()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W.setdefault(i, dict())
                W[i].setdefault(j, 0)
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W
    
    # 计算物品相似度 ItemIUF
    def ItemSimilarityItemIUF(train):
        # calculate co-rated users between items
        C = dict()
        N = dict()
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    C.setdefault(i, dict())
                    C[i].setdefault(j, 0)
                    C[i][j] += 1 / math.log(1 + len(items) * 1.0)
        
        # calculate finial similarity matrix W
        W = dict()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W.setdefault(i, dict())
                W[i].setdefault(j, 0)
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W

    # 计算用户历史上喜欢的物品为现在的推荐结果进行解释
    def Recommend(train, user_id, W, K, N):
        rank = dict()
        ru = train[user_id]
        for i, pi in ru.items():
            for j, wj in sorted(W[i].items(), key = itemgetter(1), reverse = True)[0:K]:
                if j in ru:
                    continue
                rank.setdefault(j, 0)
                rank[j] += pi * wj
        return sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]

    # 计算召回率
    def Recall(train, test, W, K, N):
        hit = 0
        all = 0
        for user in train.keys():
            tu = test.get(user, dict())
            rank = ItemBasedCF.Recommend(train, user, W, K, N)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += len(tu)
        return hit / (all * 1.0)

    # 计算准确率
    def Precision(train, test, W, K, N):
        hit = 0
        all = 0
        for user in train.keys():
            tu = test.get(user, dict())
            rank = ItemBasedCF.Recommend(train, user, W, K, N)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += N
        return hit / (all * 1.0)

    # 计算覆盖率
    def Coverage(train, test, W, K, N):
        recommend_items = set()
        all_items = set()
        for user in train.keys():
            for item in train.get(user, dict()).keys():
                all_items.add(item)
            rank = ItemBasedCF.Recommend(train, user, W, K, N)
            for item, pui in rank:
                recommend_items.add(item)
        return len(recommend_items) / (len(all_items) * 1.0)

    # 计算新颖度
    def Popularity(train, test, W, K, N):
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                if item not in item_popularity:
                    item_popularity.setdefault(item, 0)
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        for user in train.keys():
            rank = ItemBasedCF.Recommend(train, user, W, K, N)
            for item, pui in rank:
                ret += math.log(1 + item_popularity[item])
                n += 1
        ret /= n * 1.0
        return ret


if __name__ == '__main__':
    # 将 u.data 文件转成 u.csv 文件
    if not os.path.exists('u_ratings.csv'):
        u = np.loadtxt('u.data')
        uDataFrame = pd.DataFrame(u, columns = [ 'userId', 'movieId', 'rating', 'timestamp' ])
        uDataFrame.to_csv('u_ratings.csv', index = False)
    
    data = pd.read_csv('u_ratings.csv')
    train, test = ItemBasedCF.SplitData(data, 8, 1, 0)
    # W = ItemBasedCF.ItemSimilarityItemCF(train)
    W = ItemBasedCF.ItemSimilarityItemIUF(train)

    K = 80
    N = 10
    print('召回率：', ItemBasedCF.Recall(train, test, W, K, N))
    print('准确率：', ItemBasedCF.Precision(train, test, W, K, N))
    print('覆盖率：', ItemBasedCF.Coverage(train, test, W, K, N))
    print('新颖度：', ItemBasedCF.Popularity(train, test, W, K, N))
    