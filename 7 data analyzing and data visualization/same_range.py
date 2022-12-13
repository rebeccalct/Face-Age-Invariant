# generate within-age cluster
import pandas as pd
import numpy as np
result_txt_path = "C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\result\\deepface.txt"
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
        ps.append(p)
        p1.append(pic1.split("-")[1])
        p2.append(pic2.split("-")[1])
        sims.append(sim)

df = pd.DataFrame(list(zip(ps, p1, p2, sims)), columns =["Person",'Picture1', 'Picture2',"Similarity"]) 

# Within same age range
per_s = list(set(ps)) # remove duplicate number 
sum = pd.DataFrame(columns=("Person","<=10","10-20","20-30","30-40","40-50","50-60","60-70","70+"))
sum['Person'] = pd.Series(per_s)

i=0
for p in per_s:
    df_p = df[df["Person"]== p]
    age_list1 = df_p["Picture1"].apply(lambda x: float(x))
    age_list2 = df_p["Picture2"].apply(lambda x: float(x))

    # different age range
    df1 = df_p.loc[(age_list1<=10)&(age_list2<=10)]
    df2 = df_p.loc[(age_list1>10)&(age_list2>10)&(age_list1<=20)&(age_list2<=20)]
    df3 = df_p.loc[(age_list1>20)&(age_list2>20)&(age_list1<=30)&(age_list2<=30)]
    df4 = df_p.loc[(age_list1>30)&(age_list2>30)&(age_list1<=40)&(age_list2<=40)]
    df5 = df_p.loc[(age_list1>40)&(age_list2>40)&(age_list1<=50)&(age_list2<=50)]
    df6 = df_p.loc[(age_list1>50)&(age_list2>50)&(age_list1<=60)&(age_list2<=60)]
    df7 = df_p.loc[(age_list1>60)&(age_list2>60)&(age_list1<=70)&(age_list2<=70)]
    df8 = df_p.loc[(age_list1>70)&(age_list2>70)]


    sum.loc[i,"<=10"] = df1["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"10-20"]= df2["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"20-30"] = df3["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"30-40"] = df4["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"40-50"] = df5["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"50-60"] = df6["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"60-70"] = df7["Similarity"].apply(lambda x: float(x)).mean()
    sum.loc[i,"70+"] = df8["Similarity"].apply(lambda x: float(x)).mean()
    print(i)

    i += 1 

sum.to_csv('C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\result\\deepface1.csv')

# pairs.txt has some problem because of my previous don't deal with all picture name
