# Face-Age-Invariant
Project in data science ( 1DL505, UPPSALA, Prof. Anders Hast)

## Data source
Currently we are using AgeDB datasetthe first manually collected, in-the-wild age database, including almost Europeans or some Africans, almost no Asian faces. 

### Data cleaning
From raw datasets, we noticed the label of some pictures are wrong. So we have to revise them or delete them directly.
1. wrong "gender"
2. more than one face in the picture

### Picture re-organization
The dataset is sorted to get the serial number of each person. Every picture is renamed by the age.jpg.
1. sort by name, get the number of pictures in every age range;
2. filter the "total pictures" less than 8
3. change the file  name

### Generate comparisons.txt



