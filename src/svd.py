import numpy as np
from math import sqrt
import pickle
import time
import csv
import random
# douban: 2343  83023
# imdb: 4076 48647
douban_shape = [2343, 83023]
imdb_shape = [4076, 48647]
movielens_sharp = [943, 1682]
shape = movielens_sharp
def load_moviesles_data():
    train_data = {}
    test_data = {}
    with open("train.dat", 'r', encoding="utf-8") as f:
        for line in f:
            row = line.split("\t")
            movie_index = int(row[0])-1
            user_index = int(row[1])-1
            score = float(row[2])
            if movie_index in train_data:
                train_data[movie_index][user_index] = score
            else:
                train_data[movie_index] = {}
                train_data[movie_index][user_index] = score
    with open("test.dat", 'r', encoding="utf-8") as f:
        for line in f:
            row = line.split("\t")
            movie_index = int(row[0])
            user_index = int(row[1])
            score = float(row[2])
            if movie_index in test_data:
                test_data[movie_index][user_index] = score
            else:
                test_data[movie_index] = {}
                test_data[movie_index][user_index] = score
    return train_data, test_data

def load_data(file_name):
    train_data = {}
    test_data = {}
    # 数据存储以字典形式
    # 字典有两层，外层的key为movieid,内层的为userid
    # 取值的时候就是data[movieid][userid]
    with open(file_name, 'r', encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            i = random.randint(1,10)
            movie_index = int(row[0])
            user_index = int(row[1])
            score = float(row[2])
            if i <= 8:
                if movie_index in train_data:
                    train_data[movie_index][user_index] = score
                else:
                    train_data[movie_index] = {}
                    train_data[movie_index][user_index] = score
            else:
                if movie_index in test_data:
                    test_data[movie_index][user_index] = score
                else:
                    test_data[movie_index] = {}
                    test_data[movie_index][user_index] = score
    return train_data, test_data

def save(obj, file_name):
    output = open(file_name, 'wb')
    pickle.dump(obj, output)
    output.close()

def svd(data, feature, steps=500, gama=0.02, lamda=0.3):
    slowRate = 0.99
    preRmse = 1000000000.0
    nowRmse = 0.0
    item_feature = np.matrix(np.random.rand(shape[0], feature), dtype=np.longfloat) #longfloat 一定程度上减少了溢出的可能性
    user_feature = np.matrix(np.random.rand(shape[1], feature), dtype=np.longfloat)
    print("create random user and item feature matrix success.")
    for step in range(steps):
        rmse = 0.0
        n = 0
        print("start step %d" % (step+1))
        for i in data:# item即moviez在外层字典
            for u in data[i]: # uesr在内层字典中
                pui = float(np.dot(item_feature[i,:], user_feature[u,:].T))
                eui = data[i][u] - pui
                rmse += pow(eui, 2)
                for k in range(feature):
                    user_feature[u,k] += gama*(eui*item_feature[i,k] - lamda*user_feature[u,k])
                    item_feature[i,k] += gama*(eui*user_feature[u,k] - lamda*item_feature[i,k]) # 原blog这里有错误
                n += 1
        nowRmse = sqrt(rmse * 1.0 / n)
        print ('step: %d  Rmse: %s' % ((step+1), nowRmse))
        if (nowRmse < preRmse):
            preRmse = nowRmse
        else:
            break # 这个退出条件其实还有点问题
        gama *= slowRate
        step += 1
    return user_feature, item_feature

def test(user_feature, item_feature, test_data):
    rmse = 0.0
    n = 0
    for i in test_data:
        for u in test_data[i]:
            pui = float(np.dot(item_feature[i, :], user_feature[i, :].T))
            eui = test_data[i][u] - pui
            rmse += pow(eui, 2)
            n += 1
    nowRmse = sqrt(rmse * 1.0 / n)
    return nowRmse

def main():
    starttime = time.clock()
    # train_data, test_data  = load_data("E:/Data/Movie/douban/movie_score.csv")
    train_data, test_data = load_moviesles_data()
    save(test_data, "test_data.pkl")
    save(train_data, "train_data.pkl")
    print("load data success.")

    user_feature, item_feature = svd(train_data, 100, steps=50, gama=0.02, lamda=0.15)
    save(user_feature, "user_feature.pkl")
    save(item_feature, "item_feature.pkl")
    print("svd success.")
    print("the rmse in test set : %s." % (test(user_feature, item_feature, test_data)))
    endtime = time.clock()
    runtime = endtime - starttime
    print("the Runtime is: %s." % runtime)

if __name__ == "__main__":
    main()