import numpy as np
import os
import pandas as pd
import random
import math
from operator import itemgetter

class UserBasedCF:
    
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

    # 计算用户相似度 UserCF
    def UserSimilarityUserCF(train):
        # build inverse table for item_users
        item_users = dict()
        for u, items in train.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        # calculate co-rated items between users
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    C.setdefault(u, dict())
                    C[u].setdefault(v, 0)
                    C[u][v] += 1

        # calculate finial similarity matrix W
        W = dict()
        for u, related_users in C.items():
            for u, cuv in related_users.items():
                W.setdefault(u, dict())
                W[u].setdefault(v, 0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W

    # 计算用户相似度 User-IIF
    def UserSimilarityUserIIF(train):
        # build inverse table for item_users
        item_users = dict()
        for u, items in train.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)
        
        # calculate co-rated items between users
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                for v in users:
                    if u == v:
                        continue
                    C.setdefault(u, dict())
                    C[u].setdefault(v, 0)
                    C[u][v] += 1 / math.log(1 + len(users))
        
        # calculate finial similarity matrix W
        W = dict()
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                W.setdefault(u, dict())
                W[u].setdefault(v, 0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W
    
    # 为每个用户选出 K 个和他兴趣最相似的用户，然后推荐 N 个那 K 个用户感兴趣的物品
    def Recommend(user, train, W, K, N):
        rank = dict()
        interacted_items = train[user]
        for v, wuv in sorted(W[user].items(), key = itemgetter(1), reverse = True)[0:K]:
            for i, rvi in train[v].items():
                if i in interacted_items:
                    # we should filter items user interacted before
                    continue
                rank.setdefault(i, 0)
                rank[i] += float(wuv) * float(rvi)
        return sorted(rank.items(), key = itemgetter(1), reverse = True)[0:N]
    
    # 计算召回率
    def Recall(train, test, W, K, N):
        hit = 0
        all = 0
        for user in train.keys():
            tu = test.get(user, dict())
            rank = UserBasedCF.Recommend(user, train, W, K, N)
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
            rank = UserBasedCF.Recommend(user, train, W, K, N)
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
            rank = UserBasedCF.Recommend(user, train, W, K, N)
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
            rank = UserBasedCF.Recommend(user, train, W, K, N)
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
    train, test = UserBasedCF.SplitData(data, 8, 1, 0)
    # W = UserBasedCF.UserSimilarityUserCF(train)
    W = UserBasedCF.UserSimilarityUserIIF(train)

    # 为每个用户选出 80 个和他兴趣最相似的用户，然后推荐 10 个那 80 个用户感兴趣的物品
    K = 80
    N = 10
    print('召回率：', UserBasedCF.Recall(train, test, W, K, N))
    print('准确率：', UserBasedCF.Precision(train, test, W, K, N))
    print('覆盖率：', UserBasedCF.Coverage(train, test, W, K, N))
    print('新颖度：', UserBasedCF.Popularity(train, test, W, K, N))
    