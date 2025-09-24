import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns


def nedostaje(df, provera):
    for kolona in provera:
        print(kolona, df[kolona].isna().sum(), "of", len(df[kolona]))
def age_dist(df):
    plt.figure(figsize=(12, 8))
    plt.hist(df['Age'], bins=50, edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('age_dist.png')
    plt.close()
def dists(df):
    plt.figure(figsize=(12, 8))
    plt.hist(df['Pclass'], bins=3, edgecolor='black')
    plt.title('class Distribution')
    plt.xlabel('class')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('class_dist.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.hist(df['Sex'], bins=2, edgecolor='black')
    plt.title('sex Distribution')
    plt.xlabel('sex')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('gender_dist.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.hist(df['SibSp'], bins=10, edgecolor='black')
    plt.title('sib Distribution')
    plt.xlabel('sib')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('sibsp_dist.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.hist(df['Parch'], bins=10, edgecolor='black')
    plt.title('parch Distribution')
    plt.xlabel('parch')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('parch_dist.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.hist(df['Fare'], bins=20, edgecolor='black')
    plt.title('fare Distribution')
    plt.xlabel('fare')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('fare_dist.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.hist(df['Embarked'], bins=3, edgecolor='black')
    plt.title('port Distribution')
    plt.xlabel('port')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig('port_dist.png')
    plt.close()
def AttCorrelation(df):
    korelacije = df.corr()
    # print(korelacije)
    sns.heatmap(korelacije, annot=True, cmap='coolwarm')
    plt.savefig('attribute_correlation.png')
    plt.close()
def targetCorrelation(X, y):
    Xy = pd.DataFrame(X)
    Xy['Survived'] = y

    cor = Xy.corr()[['Survived']].drop('Survived')
    plt.figure(figsize=(12, 8))
    sns.heatmap(cor, annot=True, cmap='coolwarm')
    plt.savefig('target_correlation.png')
    plt.close()
def survivedby(X, y):
    Xy = pd.DataFrame(X)
    Xy['Survived'] = y

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='Survived', data=Xy)
    plt.title("Survival by Sex")
    plt.tight_layout()
    plt.savefig("output/eda_survival_by_sex.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Pclass', hue='Survived', data=Xy)
    plt.title("Survival by Class")
    plt.tight_layout()
    plt.savefig("output/eda_survival_by_class.png")
    plt.close()

    plt.figure(figsize=(20, 8))
    sns.countplot(x='Age', hue='Survived', data=Xy)
    plt.title("Survival by Age")
    plt.savefig("output/eda_survival_by_age.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='SibSp', hue='Survived', data=Xy)
    plt.title("Survival by sibsp")
    plt.tight_layout()
    plt.savefig("output/eda_survival_by_sibsp.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Parch', hue='Survived', data=Xy)
    plt.title("Survival by parch")
    plt.tight_layout()
    plt.savefig("output/eda_survival_by_parch.png")
    plt.close()

    plt.figure(figsize=(20, 8))
    sns.countplot(x='Fare', hue='Survived', data=Xy)
    plt.title("Survival by Fare")
    plt.savefig("output/eda_survival_by_fare.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Embarked', hue='Survived', data=Xy)
    plt.title("Survival by port")
    plt.tight_layout()
    plt.savefig("output/eda_survival_by_port.png")
    plt.close()
def fillNaN(df):
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    most_common_port = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(most_common_port)
    return df
def encode(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df


## pocetak koda

dataset = pd.read_csv('data/train.csv')
# print(dataset.head())

# deljenje podataka na data (X) i target (y)
y = dataset['Survived']
X = dataset.drop('Survived', axis=1)

# prebrojavanje koliko je nedostajucih podataka:
# kolone = ['Age', 'Cabin', 'Embarked']
# nedostaje(X, kolone)


# izbacivanje manje bitnih podataka:

# id je koristan samo ovako za polistavanje osoba, ne utice na Survived
# name isto ne utice samo po sebi, ali bi mozda bilo zanimljivo videti da li titula utice (npr ako doktori imaju veci procenat Survived od gospodina)
# karta moze biti korisna, ako bi zapravo imala neki konkretan format i/ili znacenje, mada cak i data sve sto ona moze da kaze su nam rekli klasa i cena karte
# 687 od 891 kabina fali, vrlo neupotrebljiv podatak, pored toga samo obelezje kabine ne govori tacno gde se nalazi i to bi bio posao onda istrazivati i analizirati

X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace=True)

# age_dist(X)

# iz figure se vidi da su godine normalno raspodeljene, ali postoje ekstremi, naime ima puno beba i dece, kao i poneke osobe sa preko 70 godina
# zbog toga se popunjavanje nedostajucih podataka najbolje postize upisivanjem medijane, jer na nju ne utivu ekstremi
# za embarked koristicu .mode popunjavanje, ono ce upisati najcesce u nedostajuce
X = fillNaN(X)

# kolone = ['Age', 'Embarked']
# nedostaje(X, kolone)

X = encode(X) #enkodiranje

# dists(X)
# AttCorrelation(X)
# targetCorrelation(X, y)
# survivedby(X, y)


# modeli koje koristim, u pitanju je klasifikacioni problem pa modele za to i koristim
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })
print(results)

# print(X.head())
# X.to_csv('edited.csv', index=False)
