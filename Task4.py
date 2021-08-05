import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('Cenestanova.csv')
x = dataset.iloc[:, 0:2]

#  Item 1

x = x.replace(0, np.nan)

# Item 2

x.fillna(x.mean(), inplace=True)

# Item 3

fig = plt.figure()
x1 = x.iloc[:, 1].values
y1 = x.iloc[:, 0].values
x = x.iloc[:, 0:2].values
plt.scatter(x[:, 1], x[:, 0])
plt.xlabel("Kvadratni metri")
plt.ylabel("Cene stanova")
plt.title('Cene stanova u odnosu na kvadraturu')
plt.show()
fig.savefig('4.3.png')

x1_obucavajuci, x1_testirajuci, y1_obucavajuci, y1_testirajuci = train_test_split(x1, y1, test_size=0.25,
                                                                                  random_state=1)
print(x1_obucavajuci.shape, x1_testirajuci.shape, y1_obucavajuci.shape, y1_testirajuci.shape)
regresija = LinearRegression()
x1_obucavajuci = np.reshape(x1_obucavajuci, (-1, 1))
y1_obucavajuci = np.reshape(y1_obucavajuci, (-1, 1))
x1_testirajuci = np.reshape(x1_testirajuci, (-1, 1))
y1_testirajuci = np.reshape(y1_testirajuci, (-1, 1))
print(x1_obucavajuci.shape, x1_testirajuci.shape, y1_obucavajuci.shape, y1_testirajuci.shape)
regresija.fit(x1_obucavajuci, y1_obucavajuci)
Y_predikcija = regresija.predict(x1_testirajuci)

# Item 4

print(metrics.mean_squared_error(y1_testirajuci / max(y1_testirajuci),
                                 Y_predikcija / max(Y_predikcija)))  # Srednja kvadratna greška
print(
    metrics.r2_score(y1_testirajuci / max(y1_testirajuci), Y_predikcija / max(Y_predikcija)))  # Koeficijent preklapanja

# Item 5

fig = plt.figure()
plt.scatter(x1_testirajuci, y1_testirajuci)
plt.plot(x1_testirajuci, regresija.predict(x1_testirajuci), color='red')
plt.xlabel('Kvadratni metri')
plt.ylabel('Cene stanova')
plt.title('Predviđanje cene stanova u odnosu na kvadraturu')
plt.show()
fig.savefig('4.5.png')

vrednost = 100
vrednost = np.reshape(vrednost, (-1, 1))
y_pred = regresija.predict(vrednost)
print('Cena stana od 100 kvadrata je: %.3f' % y_pred)
