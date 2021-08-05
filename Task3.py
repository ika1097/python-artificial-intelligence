import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Prisustvo_srcane_bolesti.csv')

x = dataset[dataset['klase'] == 1].iloc[:, 0:2].values
y = dataset[dataset['klase'] == 2].iloc[:, 0:2].values

# Item 1.

fig = plt.figure()
plt.plot(x[:, 0], x[:, 1], 'cx', label='Odsustvo srčane bolesti')
plt.plot(y[:, 0], y[:, 1], 'y+', label='Prisustvo srčane bolesti')
plt.xlabel('prvo obeležje klase')
plt.ylabel('drugo obeležje klase')
plt.legend()
plt.title('Parametri za prisustvo i odsustvo srčane bolesti u 2D prostoru')
plt.show()
fig.savefig('3.1.png')

# Item 2.

z = np.arange(-10, 10, 0.1)
u1 = norm.pdf(z, np.mean(x[:, 0]), np.std(x[:, 0]))
u2 = norm.pdf(z, np.mean(x[:, 1]), np.std(x[:, 1]))
fig = plt.figure()
plt.plot(z, u1, 'r-', label='x, prvo')
plt.plot(z, u2, 'k-.', label='x, drugo')
plt.grid(True)
plt.legend()
plt.title('Gustine raspodele za klasu odsustvo srčane bolesti')
plt.show()
fig.savefig('3.2.1.png')

z = np.arange(-10, 10, 0.1)
o1 = norm.pdf(z, np.mean(y[:, 0]), np.std(y[:, 0]))
o2 = norm.pdf(z, np.mean(y[:, 1]), np.std(y[:, 1]))
fig = plt.figure()
plt.plot(z, o1, 'r-', label='y prvo')
plt.plot(z, o2, 'k-.', label='y drugo')
plt.grid(True)
plt.legend()
plt.title('Gustine raspodele za klasu prisustvo srčane bolesti')
plt.show()
fig.savefig('3.2.2.png')

z = np.arange(-10, 10, 0.1)
u1 = norm.pdf(z, np.mean(x[:, 0]), np.std(x[:, 0]))
o1 = norm.pdf(z, np.mean(y[:, 0]), np.std(y[:, 0]))
fig = plt.figure()
plt.plot(z, u1, 'r-', label='odsustvo, prva')
plt.plot(z, o1, 'k-.', label='prisustvo, prva')
plt.grid(True)
plt.legend()
plt.title('Gustine raspodele prvo obelezje klasa Odsustvo i prisustvo srčane bolesti')
plt.show()
fig.savefig('3.2.3.png')

z = np.arange(-10, 10, 0.1)
u2 = norm.pdf(z, np.mean(x[:, 1]), np.std(x[:, 1]))
o2 = norm.pdf(z, np.mean(y[:, 1]), np.std(y[:, 1]))
fig = plt.figure()
plt.plot(z, u2, 'r-', label='odsustvo, druga')
plt.plot(z, o2, 'k-.', label='prisustvo, druga')
plt.grid(True)
plt.legend()
plt.title('Gustine raspodele drugo obelezje klasa Odsustvo i prisustvo srčane bolesti')
plt.show()
fig.savefig('3.2.4.png')

# Item 3.

X = dataset.iloc[:, 0:2].values
Y = dataset.iloc[:, 2].values
X_obucavajuci, X_testirajuci, Y_obucavajuci, Y_testirajuci = train_test_split(X, Y, test_size=0.20, random_state=1)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_obucavajuci, Y_obucavajuci, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))  # štampamo srednju vrednost tačnosti
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
fig = plt.figure()

ax = sns.boxplot(data=results)
ax = sns.swarmplot(data=results, color="black")
plt.xticks(np.arange(0, 4), names)
plt.title('Grafik klasifikatora')
plt.show()
fig.savefig('3.3.png')

# Item 4. i 5.

model = GaussianNB()

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model.fit(X_obucavajuci, Y_obucavajuci)
predictions = model.predict(X_testirajuci)
print(accuracy_score(Y_testirajuci, predictions))
print(confusion_matrix(Y_testirajuci, predictions))
print(classification_report(Y_testirajuci, predictions))

import mlxtend
from mlxtend.plotting import plot_decision_regions

fig = plt.figure()
plot_decision_regions(X_obucavajuci, Y_obucavajuci.astype(np.integer), clf=model)
plt.title('Granice odluke za obučavajući skup')
plt.show()
fig.savefig('3.5.1.png')

fig = plt.figure()
plot_decision_regions(X_testirajuci, Y_testirajuci.astype(np.integer), clf=model, legend=2)
plt.title('Granice odluke za testirajući skup')
plt.show()
fig.savefig('3.5.3.png')
