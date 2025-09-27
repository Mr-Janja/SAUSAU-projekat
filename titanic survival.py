import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
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
def findconfusionmatrix(y_test, y_pred, name, test_size):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['did not Survive', 'Survived'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('output/cm/' + name + f'{test_size}.png')
    plt.close()
def findBestModelAndTestSize(models, X, y, test_sizes, results):
    best = {'LogisticRegression': 0, 'DecisionTree': 0, 'RandomForest': 0, 'KNeighborsClassifier': 0}
    bestts = {'LogisticRegression': 0, 'DecisionTree': 0, 'RandomForest': 0, 'KNeighborsClassifier': 0}
    for name, model in models.items():
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # findconfusionmatrix(y_test, y_pred, name, test_size)
            weighted_score =0.1 * accuracy_score(y_test, y_pred) + 0.2 * precision_score(y_test,y_pred) + 0.35 * recall_score(y_test, y_pred) + 0.35 * f1_score(y_test, y_pred)
            results.append({
                "Model": name,
                "Test Size": test_size,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "average score": weighted_score
            })
            if weighted_score > best[name]:
                best[name] = weighted_score
                bestts[name] = test_size
    # recall i f1 score su mi najbolje metrike, precision je potreban ako imam veliki trosak u pogresnom pogadjanju prezivelih, average je manje bitan jer moze omanuti model (dataset je nebalansiran, vise ljudi je umrlo nego prezivelo, pa ako bi model rekao da su svi preminuli to moze dati veliki accuracy a to nam ne treba)
    # odredjivanje najboljeg modela po weighted score-u
    # print(best)
    # print(bestts)
    bestkey = max(best, key=best.get)
    # print(f"najbolji model po average weighted score-u je: {bestkey} : {best[bestkey]} za test_size = {bestts[bestkey]} ")
    return bestkey, bestts[bestkey]
def unakrsnaValidacija(X, y, ts, model, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, stratify=y, random_state=42)

    gs = GridSearchCV(model, param_grid=param_grid, cv=10, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
    # print("best parameters")
    # print(gs.best_params_)
    modeltree = gs.best_estimator_
    print(modeltree)
    y_pred = modeltree.predict(X_test)
    resultstree = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "average score": 0.1 * accuracy_score(y_test, y_pred) + 0.2 * precision_score(y_test,y_pred) + 0.35 * recall_score(y_test, y_pred) + 0.35 * f1_score(y_test, y_pred)
    }
    print("DecisionTree results:")
    print(resultstree)
    return modeltree
def findFeatureImportance(model, X, y, ts):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_names = X.columns
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df.sort_values(by='Importance', ascending=False, inplace=True)
    plt.figure(figsize=(12, 8))
    plt.barh(imp_df['Feature'], imp_df['Importance'])
    plt.title('Feature Importance')
    plt.savefig('output/feature_importance.png')
    plt.close()
    # print(imp_df)
def modelWithLessAttributes(X, y, ts, modelCV):
    X_new = X.drop(['SibSp', 'Parch', 'Embarked'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=ts, random_state=42)
    modelCV.fit(X_train, y_train)
    y_pred = modelCV.predict(X_test)
    weighted_score = 0.1 * accuracy_score(y_test, y_pred) + 0.2 * precision_score(y_test, y_pred) + 0.35 * recall_score(y_test, y_pred) + 0.35 * f1_score(y_test, y_pred)
    print("DecisionTree results with removed attributes:")
    print(accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), weighted_score)

###############################################
###############################################
##############  MAIN ##########################
###############################################
###############################################

dataset = pd.read_csv('data/train.csv')
# deljenje podataka na data (X) i target (y)
y = dataset['Survived']
X = dataset.drop('Survived', axis=1)

# prebrojavanje koliko je nedostajucih podataka:
kolone = ['Age', 'Cabin', 'Embarked']
nedostaje(X, kolone)

# izbacivanje manje bitnih podataka:

# id je koristan samo ovako za polistavanje osoba, ne utice na Survived
# name isto ne utice samo po sebi, ali bi mozda bilo zanimljivo videti da li titula utice (npr ako doktori imaju veci procenat Survived od gospodina)
# karta moze biti korisna, ako bi zapravo imala neki konkretan format i/ili znacenje, mada cak i data sve sto ona moze da kaze su nam rekli klasa i cena karte
# 687 od 891 kabina fali, vrlo neupotrebljiv podatak, pored toga samo obelezje kabine ne govori tacno gde se nalazi i to bi bio posao onda istrazivati i analizirati

X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace=True)

age_dist(X)

# iz figure se vidi da su godine normalno raspodeljene, ali postoje ekstremi, naime ima puno beba i dece, kao i poneke osobe sa preko 70 godina
# zbog toga se popunjavanje nedostajucih podataka najbolje postize upisivanjem medijane, jer na nju ne utivu ekstremi
# za embarked koristicu .mode popunjavanje, ono ce upisati najcesce u nedostajuce
X = fillNaN(X)
X = encode(X) #enkodiranje
X.to_csv('edited.csv', index=False)

dists(X)
AttCorrelation(X)
targetCorrelation(X, y)
survivedby(X, y)


# modeli koje koristim, u pitanju je klasifikacioni problem pa modele za to i koristim
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=0),
    'RandomForest': RandomForestClassifier(random_state=0),
    'KNeighborsClassifier': KNeighborsClassifier(),
}
results = []
test_sizes = {0.1, 0.2, 0.25, 0.3}
model, ts = findBestModelAndTestSize(models, X, y, test_sizes, results)
# best: decision tree, test_size 0.2

param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'ccp_alpha': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
}
modelForCV = models[model]
modelCV = unakrsnaValidacija(X, y, ts, modelForCV, param_grid)

findFeatureImportance(modelCV, X, y, ts)
modelWithLessAttributes(X, y, ts, modelCV)


