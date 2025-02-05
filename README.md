<h1 align="center"> Rational or Emotional? Next-item Recommendations in
Virtual Games via Disentangling Players’ Needs </h1>


## About Our Work

Update: 2025/02/12: We have created a repository for the paper titled *Rational or Emotional? Next-item Recommendations in Virtual Games via Disentangling Players’ Needs*, which has been submitted to the *Journal on Computing*. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![image-20240904095949527](https://s2.loli.net/2024/09/04/1jcyXSpGCiOkQNT.png)



## Requirements

- torch==1.13.0+cu117
- dgl==1.1.2
- scikit-learn==1.13.2
- seaborn==0.13.0
- x-transformers==1.34.0

## Data Sets

We have uploaded the sample dataset of netease, Amazon-Video Games, and Food. For Amazon VideoGames, you could download the file in http://jmcauley.ucsd.edu/data/amazon/links.html.

The structure of the data set should be like,

```powershell
data
|_ raw
|  |_ food
|  |_ _ Interactions_train.csv
|  |_ _***
|  |_ netease
|  |_ _ role_info.json
|  |_ _ item_info.json
|  |_ _ 1.csv
|  |_ _***
|  |_ video
|  |_ _ meta_Video_Games.json.gz
|  |_ _***
|_ ready
|  |_ food
|  |_ _ CXT_food.dat
|  |_ _ CXT_food.txt
|  |_ _ food.txt
|  |_ _ food_cxt.txt
|  |_ _ food_fea_dat
|  |_ ***
```

## RUN

```powershell
# process the data
python data_pro.py
# run the file
python main.py
```

## Acknowledge & Contact

We will update the contact information and corresponding issue links after the review process is completed.
