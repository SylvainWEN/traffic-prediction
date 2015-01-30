__author__ = 'wenjf'

import lib_wordsList
import json
import csv
import re
import time
import nltk
import numpy as np
from sklearn import linear_model
import os

# all time is represented by time_start + gap_time
time_start = 'Sat Mar 01 00:00:00 2014'
stamp_start = time.mktime(time.strptime(time_start, "%a %b %d %H:%M:%S %Y"))

#calculationg for de-trended
de_trended = []
for i in range(0, 7):
    hours = []
    for j in range(0, 24):
        hours.append([])
    de_trended.append(hours)

# data of gps averaged over all
list_gps = {}
# data of twitter in DC
list_twit = {}

list_t_gps = {}

#dict for processing twitter
# all_vocab = set(w.lower() for w in nltk.corpus.words.words())
# stop_w = set(lib_wordsList.stopwords)
# vocab = list(all_vocab.difference(stop_w))
vocab = lib_wordsList.trafficBadKW + lib_wordsList.trafficGoodKW
dim_vec = len(vocab)


# read twitter data from one file
def one_hour_twit(path):
    test_file = open(path, 'r', encoding='utf-8')
    for L in test_file:
        js = json.loads(L)
        # print(js['created_at'])
        this_twitter = js['text']
        twiter_word = set(re.split(r'[ \t\n]+', this_twitter))
        # print(twiter_word)
        twit_vec = []
        for i in range(0, dim_vec):
            if vocab[i] in twiter_word:
                twit_vec.append(1)
            else:
                twit_vec.append(0)
        twit_vec.append(1)
        twit_vec = np.array(twit_vec)
        print(twit_vec)
        pre_treat = js['created_at'].replace('+0000', '')
        t_to_start = time.mktime(time.strptime(pre_treat, "%a %b %d %H:%M:%S %Y"))
        #当前时间变成时间戳
        t_stamp = 0
        if not (t_to_start-stamp_start) % 300 == 0:
            t_stamp = stamp_start + 300 * ((t_to_start-stamp_start)//300) + 300
        time_twit = time.mktime(time.localtime(t_stamp))
        if time_twit in list_twit:
            list_twit[time_twit].append(twit_vec)
        else:
            list_twit[time_twit] = []
            list_twit[time_twit].append(twit_vec)



def twit_to_vect(twitterZ_here):
    return 0

def get_gps_data(path):
    import time as time_here
    reader = csv.reader(open(path, 'r'))
    # ['tmc_code', 'measurement_tstamp', 'speed', 'average_speed', 'reference_speed', 'travel_time_minutes', 'confidence_score', 'cvalue']
    for line in reader:
        t_stamp = earlier(line[1])
        # if t_stamp > 20140308000000:
        if t_stamp > 20140302000000:
            break
        if t_stamp == 0:
            continue
        speed = float(line[2])
        t_stamp = str(t_stamp)
        # print(t_stamp)
        t_stamp_format = t_stamp[0:4]+'-'+t_stamp[4:6]+'-'+t_stamp[6:8]+' '+t_stamp[8:10]+':'+t_stamp[10:12] + ':' + t_stamp[12:]
        t_stamp_final = time.mktime(time.strptime(t_stamp_format, '%Y-%m-%d %H:%M:%S'))
        if not t_stamp_final in list_t_gps:
            list_t_gps[t_stamp_final] = 1
            list_gps[t_stamp_final] = speed
        else:
            list_t_gps[t_stamp_final] += 1
            list_gps[t_stamp_final] += speed

        # list_gps_ori.append(one_gps(t_stamp, speed))

        # if t_stamp == 0:
        #     continue
        # if t_stamp > 20140301235500:
        #     break
        # if t_stamp == old_time:
        #     speed_average = speed_average + float(line[2])
        #     number += 1
        #     continue
        # else:
        #     # print(old_time)
        #     old_time = t_stamp
        #     # print(t_stamp)
        #     print(number)
        #     number = 1
        #     one_ele = one_gps(t_stamp-500, speed_average/number)
        #     list_gps.append(one_ele)

def earlier(time_chosen):
    pattern = re.compile(r'\d+')
    result = pattern.findall(time_chosen)
    res = ''
    if len(result) > 1:
        for ele in result:
            res += ele
        return int(res)
    else:
        return 0

def list_average(para_list):
    size = len(para_list)
    sum = 0
    for each_speed in para_list:
        sum += each_speed
    averaged_speed = 0
    if not size == 0:
        averaged_speed = sum/size
    return averaged_speed

#  in fact we can process the raw data in constructor of one_gps, but I do a pre-processing in earlier.

# class one_gps:
#     def __init__(self, time, speed):
#         self.t_stamp = time
#         self.t_speed = speed
#     def test(self):
#         print(self.t_stamp)
#         print(self.t_speed)

# class one_twitter:
#     def __init__(self, time, content):
#         self.t_stamp = time
#         self.text = content
#     def test(self):
#         print(self.t_stamp)
#         print(self.text)

all_file = os.listdir('E:\\ACT\\投稿\\KDDforLINLU\\KDD\\ICJAI')
for destination in all_file:
    dest_final = 'E:\\ACT\\投稿\\KDDforLINLU\\KDD\\ICJAI\\' + destination
    one_hour_twit(dest_final)
    # print(destination)
    # print(len(list_twit))

# all_time_t = list_twit.keys()
# for time_here in all_time_t:
#     print(time_here)
#     print(list_twit[time_here])

#begining of processing gps_data
get_gps_data('E:\\ACT\\投稿\\KDDforLINLU\\KDD\\Readings.csv')
all_time = list(list_t_gps.keys())

all_time.sort()
# output_no_de = open('.\\no_de.txt', 'w')
for L in range(0, len(all_time)):
    list_gps[all_time[L]] = list_gps[all_time[L]] / list_t_gps[all_time[L]]
#     output_no_de.write(str(list_gps[all_time[L]])+'\t')
# output_no_de.close()
#end of processing gps_data
#begining of processing gps_data de-trended
for time_here in all_time:
    time_dh = time.localtime(time_here)
    day = time_dh[6]
    hour = time_dh[3]
    de_trended[day][hour].append(list_gps[time_here])
for i in range(0, 7):
    for j in range(0, 24):
        # print(de_trended[i][j])
        de_trended[i][j] = list_average(de_trended[i][j])
        # print(de_trended[i][j])
for time_here in all_time:
    time_dh = time.localtime(time_here)
    day = time_dh[6]
    hour = time_dh[3]
    list_gps[time_here] = list_gps[time_here] - de_trended[day][hour]
    # print(list_gps[time_here])
#end of processing gps_data de_trended

#plot of gps_data

# output = open('.\\plot.txt', 'w', encoding='utf-8')
# for L in range(0, len(all_time)):
#     output.write(str(list_gps[all_time[L]])+'\n')
# output.close()
#end of plot of gps_data

#end of data processing


# for time_here in all_time:
#     print('gps_time')
#     print(time_here)
time_twit = list_twit.keys()
for time_here in all_time:
    if time_here in list_twit:
        list_twit[time_here] = np.array(list_twit[time_here])
        # print(list_twit[time_here].shape)


#begining of solving equations

#init parameters
m=list()
for i in range(0, dim_vec):
    m.append(0)
m.append(1)
m = np.array(m)
m = m.reshape((-1, 1))
alpha = 0
beta_1 = 0
beta_2 = 0
gamma_1 = 0
gamma_2 = 0
num_time_stamp = len(all_time)


def linear_re():
    lr_matrix = []
    lr_traget = []
    for i in range(3, num_time_stamp):
        one_row = list()
        one_row.append(list_gps[all_time[i-1]])
        one_row.append(list_gps[all_time[i-2]])
        if all_time[i-1] in time_twit:
            size = list_twit[all_time[i-1]].shape[0]
            l = list()
            for j in range(0, size):
                l.append(1)
            l = np.array(l)
            l = l.reshape((1, -1))
            left_m = np.dot(l, list_twit[all_time[i-1]])
            one_row.append(np.dot(left_m, m))
        else:
            one_row.append(0)
        if all_time[i-2] in time_twit:
            size2 = list_twit[all_time[i-2]].shape[0]
            l2 = list()
            for j in range(0, size2):
                l2.append(1)
            l2 = np.array(l2)
            l2 = l2.reshape((1, -1))
            # print(l2.shape)
            # print(list_twit[all_time[i-2]].shape)
            left_m = np.dot(l2, list_twit[all_time[i-2]])
            one_row.append(np.dot(left_m, m))
        else:
            one_row.append(0)
        lr_traget.append(list_gps[all_time[i]])
        lr_matrix.append(one_row)
    lr_matrix = np.array(lr_matrix)
    lr_traget = np.array(lr_traget)
    clf = linear_model.LinearRegression()
    clf.fit(lr_matrix, lr_traget)
    [c1, c2, c3, c4] = clf.coef_
    al = clf.intercept_
    res = list()
    res.append(al)
    res.append([c1, c2, c3, c4])
    return res

def lasso_re():
    lr_matrix = list()
    lr_traget = list()
    for i in range(3, num_time_stamp):
        one_row = list()
        for j in range(0, dim_vec+1):
            # print(j)
            one_row.append(0)
        one_row = np.array(m)
        one_row = one_row.reshape((1, -1))
        lable_here = 0
        if all_time[i-1] in time_twit:
            size = list_twit[all_time[i-1]].shape[0]
            l = list()
            for j in range(0, size):
                l.append(1)
            l = np.array(l)
            l = l.reshape((1, -1))
            one_row += gamma_1*(np.dot(l, list_twit[all_time[i-1]]))
            lable_here += 1
        if all_time[i-2] in time_twit:
            size = list_twit[all_time[i-2]].shape[0]
            l = list()
            for j in range(0, size):
                l.append(1)
            l = np.array(l)
            l = l.reshape((1, -1))
            one_row += gamma_2*(np.dot(l, list_twit[all_time[i-2]]))
            lable_here += 1
        if not lable_here == 0:
            lr_matrix.append(one_row)
            lr_traget.append(list_gps[all_time[i]] - alpha - beta_1*list_gps[all_time[i-1]] - beta_2 * list_gps[all_time[i-2]])
        else:
            continue
    clf = linear_model.Lasso(alpha=0.1, fit_intercept='false')
    lr_matrix = np.array(lr_matrix).reshape((-1, 27))
    lr_traget = np.array(lr_traget)
    print(np.array(lr_matrix).shape)
    print(lr_traget.shape)
    clf.fit(lr_matrix, lr_traget)
    res = np.array(clf.coef_).reshape((-1, 1))
    return res

mid1 = linear_re()
alpha = mid1[0]
beta_1 = mid1[1][0]
beta_2 = mid1[1][1]
gamma_1 = mid1[1][2]
gamma_2 = mid1[1][3]

print(alpha)
print(beta_1)
print(beta_2)
print(gamma_1)
print(gamma_2)

# for i in range(0, 10):
#     mid1 = linear_re()
#     alpha = mid1[0]
#     beta_1 = mid1[1][0]
#     beta_2 = mid1[1][1]
#     gamma_1 = mid1[1][2]
#     gamma_2 = mid1[1][3]
#     m = lasso_re()
# print(alpha)
# print(beta_1)
# print(beta_2)
# print(gamma_1)
# print(gamma_2)
# print(m)

# print(alpha)
# print(beta_1)
# print(beta_2)
# print(gamma_1)
# print(gamma_2)









