import numpy as np
import pandas as pd
import scipy as sc

TITANIC = pd.read_csv('train.csv', header=0,
                      names=['PassengerId', 'Survived',
                             'Pclass', 'Name', 'Sex', 'Age',
                             'SibSp', 'Parch', 'Ticket',
                             'Fare', 'Cabin', 'Embarked'])


def count_man_and_women():
    grouped = TITANIC.value_counts(subset=['Sex'])
    print('1)Количество мужчин и женщин:')
    print(grouped)

def count_ports_of_embarkion():
    grouped = TITANIC.value_counts(subset='Embarked')
    print('2) Порты посадки:')
    print(grouped)

def count_died_percent():
    people_count = len(TITANIC)
    died_count = TITANIC['Survived'].eq(0).sum()
    print(f'3)Количество погиших: {died_count}, процент: {died_count/people_count * 100:.3f} %')

def count_class_percent():
    people_count = len(TITANIC)
    first_class = TITANIC['Pclass'].eq(1).sum()
    second_class = TITANIC['Pclass'].eq(2).sum()
    third_class = TITANIC['Pclass'].eq(3).sum()
    print('4) Доли пассажиров по классам')
    print(f'1-й класс: {round(first_class/people_count*100)} %')
    print(f'2-й класс: {round(second_class/people_count*100)} %')
    print(f'3-й класс: {round(third_class/people_count*100)} %')

def pearson_sib_child():
    sib_child = TITANIC[['SibSp','Parch']]
    print('5) Корелляция между количество супругов и количеством детей')
    print(sib_child.corr('pearson'))

def pearson_survived():
    age_survived = TITANIC[['Survived', 'Age']]
    print('6) Корелляция между возрастом и выживаемостью')
    print(age_survived.corr('pearson'))

    sex_survived = TITANIC[['Survived', 'Sex']].replace('male',0).replace('female','1')
    print('Корелляция между полом и выживаемостью')
    print(sex_survived.corr('pearson'))
    
    class_survived = TITANIC[['Survived', 'Pclass']]
    print('Корелляция между классом и выживаемостью')
    print(class_survived.corr('pearson'))
    
def stats_age():
    age = TITANIC['Age']
    print('7) Статистика возраста')
    print(age.describe())

def stats_ticket():
    ticket = TITANIC['Fare']
    print('8) Статистика билетов')
    print(ticket.describe())

def most_popular_male_name():
    coltitle = (TITANIC['Name']
                .apply(lambda s: pd.Series(
                    {'Title': s.split(',')[1].split('.')[0].strip(),
                    'LastName':s.split(',')[0].strip(),
                    'FirstName':s.split(',')[1].split('.')[1].strip()})))
    joined = coltitle.join(TITANIC['Sex'])
    males_first_name = joined[joined['Sex'] == 'male']['FirstName']
    print('9)Самое популярное мужское имя на корабле')
    print(males_first_name.value_counts().head(1))

def most_popular_name_older_15():
    coltitle = (TITANIC['Name']
                .apply(lambda s: pd.Series(
                    {'Title': s.split(',')[1].split('.')[0].strip(),
                    'LastName':s.split(',')[0].strip(),
                    'FirstName':s.split(',')[1].split('.')[1].strip()})))
    joined = coltitle.join(TITANIC[['Age', 'Sex']])
    print('10) Самые популярные мужские и женские имена старше 15 лет')
    print(joined[joined['Age'] > 15].value_counts(subset=['FirstName', 'Sex']))


def main():
    count_man_and_women()
    count_ports_of_embarkion()
    count_died_percent()
    count_class_percent()
    pearson_sib_child()
    pearson_survived()
    stats_age()
    stats_ticket()
    most_popular_male_name()
    most_popular_name_older_15()

    
main()