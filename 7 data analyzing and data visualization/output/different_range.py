import pandas as pd
import numpy as np

result_txt_path = "C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\result\\paddle.txt"
p1=[]
p2=[]
sims=[]
ps=[]

file = open(result_txt_path)
while 1:
    line = file.readline().replace('\n','')
    if line == '':
        break
    line_list = line.split(', ')
    pic1,pic2,sim  = line_list[0], line_list[1], line_list[2]
    p = pic1.split("-")[0]
    a = pic1.split("-")[1]
    if "_" not in a:
        ps.append(pic1.split("-")[0])
        p1.append(pic1.split("-")[1])
        p2.append(pic2.split("-")[1])
    sims.append(sim)

df = pd.DataFrame(list(zip(ps, p1, p2, sims)), columns =["Person",'Picture1', 'Picture2',"Similarity"]) 

# different age range of same person:
import itertools
import re
age_lists = ["<=10","10-20","20-30","30-40","40-50","50-60","60-70"] # firstly leave out the 70+
comparion_lists=[]
column_names = ["Person"]
for item in itertools.combinations(age_lists,2):
    comparion_lists.append(item)
    col = item[0] + "_" + item[1]
    column_names.append(col)

plus_70_comparison=["<=10_70+","10-20_70+",'20-30_70+',"30-40_70+","40-50_70+","50-60_70+","60-70_70+"]
column_names.extend(plus_70_comparison)

per_s = list(set(ps)) # remove duplicate number 
sum2 = pd.DataFrame(columns=column_names)
sum2['Person'] = pd.Series(per_s)

'''
'<=10_10-20', '<=10_20-30', '<=10_30-40', '<=10_40-50', '<=10_50-60', '<=10_60-70'
'10-20_20-30', '10-20_30-40', '10-20_40-50', '10-20_50-60', '10-20_60-70'
'20-30_30-40', '20-30_40-50', '20-30_50-60', '20-30_60-70'
'30-40_40-50', '30-40_50-60', '30-40_60-70'
'40-50_50-60', '40-50_60-70'
'<=10_70+', '10-20_70+', '20-30_70+', '30-40_70+', '40-50_70+', '50-60_70+', '60-70_70+'
'''

i=0
for p in per_s:
    df_p = df[df["Person"]==p]
    age_list1 = df_p["Picture1"].apply(lambda x: float(x))
    age_list2 = df_p["Picture2"].apply(lambda x: float(x))

    # different age range
    j = 0 
    lenth = len(column_names)-1
    df_comparisons = [np.nan] * lenth
    for item in column_names.copy()[1:]:
        # item = column_names.copy()[1:][16]
        compare = re.findall(r"\d+",item)
        compare = [int(i) for i in compare]
        if (len(compare) == 3)&("<=10" in item.split("_")):
            age1,age2,age3= compare[0],compare[1],compare[2]
            df_comparisons[j] = df_p.loc[(age_list1<=age1)&(age_list2>age2)&(age_list1<=age3)]

        if len(compare)==4:
            age1,age2,age3,age4= compare[0],compare[1],compare[2],compare[3]
            df_comparisons[j] = df_p.loc[(age_list1>age1)&(age_list2>age3)&(age_list1<=age2)&(age_list2<=age4)]

        if "70+" in item.split("_"):
            if len(compare) == 2:
                age1,age2= compare[0],compare[1]
                df_comparisons[j] = df_p.loc[(age_list1<=age1)&(age_list2>age2)]

            else:
                age1,age2,age3= compare[0],compare[1],compare[2]
                df_comparisons[j] = df_p.loc[(age_list1>age1)&(age_list1<=age2)&(age_list2>age3)]

        j +=1

    for k  in range(lenth):
        sum2.iloc[i,k+1] = df_comparisons[k]["Similarity"].apply(lambda x: float(x)).mean()

    i +=1

sum2.to_csv('C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\result\\paddle2.csv')


