# coding=utf-8

import os
import numpy as np

np.set_printoptions(suppress=True)
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns


def dataset_mkdir(root_name):
    if not os.path.exists(root_name):
        os.mkdir(root_name)

def bias(freq, d):
    b = 0
    if 10000 < freq < 20000:
        if d <= 1:
            b = 2
        elif 1 < d <= 3:
            b = 3
        elif 3 < d <= 4:
            b = 4
    if 20000 < freq < 50000:
        if d <= 1:
            b = 1
        elif 1 < d <= 2:
            b = 2
        elif 2 < d <= 3:
            b = 3
        elif 3 < d <= 4:
            b = 4
    if 50000 < freq < 100000:
        if d <= 1:
            b = 0
        elif 1 < d <= 2:
            b = 1
        elif 2 < d <= 4:
            b = 2
    if 100000 < freq < 200000:
        if d <= 2:
            b = 0
        elif 2 < d <= 3:
            b = 1
        elif 3 < d <= 4:
            b = 2
    return b


dct = ['01', '04', '12', '15', '23', '26', '37', '45', '48', '56', '59', '67', '6a', '7b', '89', '8c', '9a', '9d', 'ab',
       'ae', 'bf', 'cd', 'de', 'ef']

pos2 = [(1, 7), (3, 7), (5, 7), (0, 6), (2, 6), (4, 6), (6, 6), (1, 5), (3, 5), (5, 5), (0, 4), (2, 4), (4, 4), (6, 4),
        (1, 3), (3, 3), (5, 3), (0, 2), (2, 2), (4, 2), (6, 2), (1, 1), (3, 1), (5, 1)]

node_index = {
    "01": "0",
    "12": "1",
    "23": "2",
    "04": "3",
    "15": "4",
    "26": "5",
    "37": "6",
    "45": "7",
    "56": "8",
    "67": "9",
    "48": "10",
    "59": "11",
    "6a": "12",
    "7b": "13",
    "89": "14",
    "9a": "15",
    "ab": "16",
    "8c": "17",
    "9d": "18",
    "ae": "19",
    "bf": "20",
    "cd": "21",
    "de": "22",
    "ef": "23"
}

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    if not os.path.exists("data_DFI_r"):
        os.mkdir("data_DFI_r")

    file_path0 = '/home/mm/Desktop/txr/pna/data_empty'
    file_path = '/home/mm/Desktop/txr/pna/data_2'

    # min_r = 0
    # dir = ''
    # empty = 0
    # full = 0

    speeds = [file_path + '/' + i for i in os.listdir(file_path)]
    for speed in speeds:
        concentration = [speed + '/' + i for i in os.listdir(speed)]
        for concen in concentration:
            for freq in os.listdir(concen):
                f = float(freq[:-4])
                front = np.load(concen + '/' + freq)[:, :2]
                data0 = np.load(file_path0 + '/' + freq)  # 对照数据
                data2 = np.load(concen + '/' + freq)[:, 5:]  # 查看的数据

                data_new = front
                n = data2.shape[1]
                for j in range(24):
                    for i in range(24):
                        d = distance(pos2[int(node_index[dct[i]])], pos2[int(node_index[dct[j]])]) / 2
                        b = bias(f, d) * 0.01/10

                        a1 = np.random.normal(0, b * data0[i, 3 * j + 1].astype('float'), 1)[0]
                        a2 = np.random.normal(0, b * data0[i, 3 * j + 1].astype('float'), 1)[0]
                        b1 = np.random.normal(0, b * 180, 1)[0]
                        b2 = np.random.normal(0, b * 180, 1)[0]

                        data0[i, 3 * j + 1] = data0[i, 3 * j + 1].astype('float') + a1
                        data2[i, 3 * j + 1] = data2[i, 3 * j + 1].astype('float') + a2
                        data0[i, 3 * j + 2] = data0[i, 3 * j + 2].astype('float') + b1
                        data2[i, 3 * j + 2] = data2[i, 3 * j + 2].astype('float') + b2

                        #print(b1,b2)
                    data_new = np.concatenate([data_new, data2[:, 3 * j:3 * j + 3]], axis=1)
                    r_diff = np.around(data0[:, 3 * j + 1].astype('float') - data2[:, 3 * j + 1].astype('float'), 5)
                    phi_diff = np.around(data0[:, 3 * j + 2].astype('float') - data2[:, 3 * j + 2].astype('float'), 5)
                    # if min(r_diff) < min_r:
                    #     min_r = min(r_diff)
                    #     dir = concen + '/' + freq + ' ' + str(j)
                    #     empty = data0[:, 3 * j + 1].astype('float')[np.argmin(r_diff)]
                    #     full = data2[:, 3 * j + 1].astype('float')[np.argmin(r_diff)]
                    data_new = np.concatenate([data_new, r_diff.reshape(-1, 1)], axis=1)
                    data_new = np.concatenate([data_new, phi_diff.reshape(-1, 1)], axis=1)

                spe = concen.split('/')[-2]
                con = concen.split('/')[-1]

                # print(min_r)
                # print(dir)
                # print(full)
                # print(empty)

                dataset_mkdir("data_DFI_r/{}".format(spe))
                dataset_mkdir("data_DFI_r/{}/{}".format(spe, con))

                # sns.kdeplot(data_new[:,4])
                # plt.hist(data_new[:,3])
                # plt.xticks(rotation = 70)
                # plt.show()
                # print(data_new)
                # #print(con+data_new[0,0]+freq)
                # #print(data_new[:,[1,3,5,6]])
                # #plt.hist(data_new[:,[1,2,3]].astype('float'))
                # plt.show()

                print(freq)
                #print(data_new[22:,112:].astype('float'))
                #print(data2[22:, 66:].astype('float'))
                #print(data0)

                np.save("data_DFI_r/{}/{}/{}".format(spe, con, freq),data_new)
