# Face-Age-Invariant
Project in data science ( 1DL505, UPPSALA, Prof. Anders Hast)

## 1. Data source
Currently we are using AgeDB datasetthe first manually collected, in-the-wild age database, including almost Europeans or some Africans, almost no Asian faces. https://ibug.doc.ic.ac.uk/resources/agedb/

### 2. Data cleaning
From raw datasets, we noticed the labels of some pictures are wrong. So we have to revise them or delete them directly.
1. remove all pictures with more than one face 
![image](https://user-images.githubusercontent.com/94723993/207734980-f538a88e-d8cf-41e4-a35a-20da56911954.png)
49 pictures are detected to have more than one face.

2. fix wrong "gender" 
![image](https://user-images.githubusercontent.com/94723993/207735666-f5df9b5f-e529-4061-ac17-5f2eb80358cb.png)
Detailed wrong "gender" information has [record.](https://github.com/rebeccalct/Face-Age-Invariant/blob/main/2%20data%20cleaning/WrongGenderInformation.pdf)
The dataset after cleaning can be download [here.](https://drive.google.com/file/d/1RFP_uNhrCpFL1fDcJ_7I_hG9WDcYjepG/view?usp=sharing)

### 3. Picture re-organization
The dataset is sorted to get the serial number of each person. Every picture is renamed by the age.jpg.
1. sort by name, get the number of pictures in every age range;
2. filter the "total pictures" less than 8
3. change the file  name

### 4. Generate comparisons.txt
In every person, we need to generate the comparions of all his/her pictures. And save this txt, it can speed up our model.

### 5. Comparisons of three models
We have applied the three models
1. insightface
2. insightface_paddle (namely paddle): Paddle_paddle is based on insightface.(https://pypi.org/project/insightface-paddle/)
3. Deepface

To generate three .txt, it is result of three models run by uppmax. And we also recorded the running time.

### 6. Uppmax
First of all, you have to request the core time by command ’interactive -A snic2022-22-1123 -n 20 -t 02:00:00’
otherwise, you will be limited because high-frequency calculations. And you have to create own virtual environment
to install some packages, otherwise access denied. To parallelize, you must def to_parallel(start)
Speed comparisons My own computer is about 60 hours, 20 cores at the same time is about 2.5 hours.

### 7. Data analyzing and Data Visualization
comparison of three methods\
1 In same age-range\
2 Age-range with other ranges

## 8. VECTOR
For vector, first we also generate the vector of all pictures(.csv) for future use. \
1.Plot the vectors of all pictures in one plot: PCA ; t-SNE\
2.Plot the vector  fo one person: PCA ; t-SNE






