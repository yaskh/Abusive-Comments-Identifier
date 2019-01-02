import pandas as pd 
import re

#Bawla Version  
"""Collects all the data points containing abusive words"""
def cleaner(data):
    data.head()
    data['comment'].str.contains('taata').astype(int).sum()
    lexicon = 'bhosda|chut|chod|chinaal|gaandu|randi|gaandfat|takke|bhenchod|bharw|khotey|lauda|kutta|taata|bdsk|bharwa|gaand|lassan|choot|maderchod|laude|ullu|chhed|bhosri|lora|kutte|bhadwe|lore|gand|randi|cuntmama|bsdk|gaandu|betichod|bhosadike|lund|rundi|bhen|kutte|hijra|chodu|chunni|jhant|dalle|tatti|mader|paad|kamina|tatay|bhsdk|gadha|bhen|kuttiya|lori|jhaat|chutiya|gandu|choot|lund|gaand|muth|gaand|moot|chut|taatay|marani|bhadhava|bhonsri|jhaant|chuda|kutti'
    abusive = data['comment'].str.contains(lexicon)
    data['rating'] = abusive
    data.head()
    abuse = data[data.rating == True]
    return abuse

lexicon = 'bhosda|chut|chod|chinaal|gaandu|randi|gaandfat|takke|bhenchod|bharw|khotey|lauda|kutta|taata|bdsk|bharwa|gaand|lassan|choot|maderchod|laude|ullu|chhed|bhosri|lora|kutte|bhadwe|lore|gand|randi|cuntmama|bsdk|gaandu|betichod|bhosadike|lund|rundi|bhen|kutte|hijra|chodu|chunni|jhant|dalle|tatti|mader|paad|kamina|tatay|bhsdk|gadha|bhen|kuttiya|lori|jhaat|chutiya|gandu|choot|lund|gaand|muth|gaand|moot|chut|taatay|marani|bhadhava|bhonsri|jhaant|chuda|kutti'
data = pd.read_csv("gaali.csv")
data = data.drop(["Unnamed: 0"] ,axis = 1) #Drops index column
data = list((set(data.comment)))
data = pd.DataFrame(data, columns = ["comment"])
abuse = cleaner(data)
data = pd.read_csv("no gaali.csv")
data = list((set(data.comment))) 

data = pd.DataFrame(data, columns = ["comment"])
abuse_1 = cleaner(data)
dataset = pd.concat([abuse,abuse_1])
del(abuse_1,abuse)
dataset.to_csv("only gaaliyan.csv", index = False)

data = data[~data.comment.str.contains(lexicon)] #Deletes all gaaliyan in no gaali dataset 
data = data[data.comment.str.len() > 30] #Drop string with len less than 30
data['rating'] = False #Creates a column with all values false


dataset = pd.concat([dataset,data])

dataset.to_csv("data.csv", index = False)

dataset.to_csv("dataset.csv")









