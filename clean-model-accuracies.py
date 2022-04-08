import re

with open("colbert-model-accuracies.txt","r") as file:
    text=file.readlines()

for i in range(len(text)):
    text[i]=re.sub('1667/1667 [==============================] - ', '', text[i])

with open("colbert-model-accuracies.txt","w") as file:
    file.writelines(text)
