import pandas as pd

def nedostaje(df, provera):
    for kolona in provera:
        print(kolona, df[kolona].isna().sum(), "of", len(df[kolona]))

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

X = encode(X) #enkodiranje

# print(X.head())
# X.to_csv('edited.csv', index=False)
