#!/usr/bin/envpython

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_data():

    base_cond=[[18,20,19,18,13,4,1],
    [20,17,12,9,3,0,0],
    [20,20,20,12,5,3,0]]
    cond1=[[18,19,18,19,20,15,14],
    [19,20,18,16,20,15,9],
    [19,20,20,20,17,10,0],
    [20,20,20,20,7,9,1]]
    cond2=[[20,20,20,20,19,17,4],
    [20,20,20,20,20,19,7],
    [19,20,20,19,19,15,2]]
    cond3=[[20,20,20,20,19,17,12],
    [18,20,19,18,13,4,1],
    [20,19,18,17,13,2,0],
    [19,18,20,20,15,6,0]]
    return base_cond,cond1,cond2,cond3

def main():
    #loadthedata
    results=get_data()

    print(results[0], len(results[0]), len(results[0][0]))
    fig=plt.figure()
    xdata = np.array(range(0, 7))
    sns.tsplot(time=xdata, data=results[0], color='r', linestyle='-')
    sns.tsplot(time=xdata, data=results[1], color='b', linestyle='--')
    sns.tsplot(time=xdata, data=results[2], color='g', linestyle='-.')
    sns.tsplot(time=xdata, data=results[3], color='k', linestyle=':')

    plt.ylabel('Sucess rate', fontsize=25)
    plt.xlabel('Iteration num', fontsize=25, labelpad=-4)
    plt.title('Robot performance', fontsize=25)
    plt.legend(loc='bottom left')
    plt.show()


if __name__=='__main__':
    main()




