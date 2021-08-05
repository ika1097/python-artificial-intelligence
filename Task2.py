import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

# Item 1

imena = ['pca1', 'pca2', 'klasa']
dataset = pd.read_csv('iris_redukovano.csv', names=imena)

s1 = dataset.iloc[:40, 0].values
s2 = dataset.iloc[:40, 1].values
s3 = dataset.iloc[:40, 2].values

y1 = dataset.iloc[50:90, 0].values
y2 = dataset.iloc[50:90, 1].values
y3 = dataset.iloc[50:90, 2].values

z1 = dataset.iloc[100:140, 0].values
z2 = dataset.iloc[100:140, 1].values
z3 = dataset.iloc[100:140, 2].values

fig = plt.figure()
plt.plot(s1, s2, 'cx', label='Setosa')
plt.plot(y1, y2, 'k*', label='Versicilor')
plt.plot(z1, z2, 'g+', label='Virginica')
plt.grid(True)
plt.title('Prikaz klasa u 2D prostoru')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('2.1.png')

# Item 2

M1 = np.array([s1.mean(), s2.mean()])
M2 = np.array([y1.mean(), y2.mean()])
M3 = np.array([z1.mean(), z2.mean()])

m11 = M1[0]
m12 = M1[1]
m21 = M2[0]
m22 = M2[1]
m31 = M3[0]
m32 = M3[1]

N = 40
setosa = dataset[dataset['klasa'] == 0]
versicolor = dataset[dataset['klasa'] == 1]
virginica = dataset[dataset['klasa'] == 2]

SigmaX = setosa.iloc[:, 0:2].cov().to_numpy()
SigmaY = versicolor.iloc[:, 0:2].cov().to_numpy()
SigmaZ = virginica.iloc[:, 0:2].cov().to_numpy()

invSigmaX = linalg.inv(SigmaX)
invSigmaY = linalg.inv(SigmaY)
invSigmaZ = linalg.inv(SigmaZ)

x11 = invSigmaX[0][0]
x12 = invSigmaX[0][1]
x21 = invSigmaX[1][0]
x22 = invSigmaX[1][1]

y11 = invSigmaY[0][0]
y12 = invSigmaY[0][1]
y21 = invSigmaY[1][0]
y22 = invSigmaY[1][1]

z11 = invSigmaZ[0][0]
z12 = invSigmaZ[0][1]
z21 = invSigmaZ[1][0]
z22 = invSigmaZ[1][1]

X = np.arange(-4, 4, 0.1)
Y = np.arange(-2, 2, 0.1)
x1, x2 = np.meshgrid(X, Y)

xy = 0.5 * (((x1 - m11) * x11 + (x2 - m12) * x21) * (x1 - m11) + ((x1 - m11) * x12 + (x2 - m12) * x22) * (
            x2 - m12)) - 0.5 * (
                 ((x1 - m21) * y11 + (x2 - m22) * y21) * (x1 - m21) + ((x1 - m21) * y12 + (x2 - m22) * y22) * (
                     x2 - m22)) + 0.5 * np.log(np.linalg.det(SigmaX) / np.linalg.det(SigmaY))
yz = 0.5 * (((x1 - m21) * y11 + (x2 - m22) * y21) * (x1 - m21) + ((x1 - m21) * y12 + (x2 - m22) * y22) * (
            x2 - m22)) - 0.5 * (
                 ((x1 - m31) * z11 + (x2 - m32) * z21) * (x1 - m31) + ((x1 - m31) * z12 + (x2 - m32) * z22) * (
                     x2 - m32)) + 0.5 * np.log(np.linalg.det(SigmaY) / np.linalg.det(SigmaZ))
xz = 0.5 * (((x1 - m11) * x11 + (x2 - m12) * x21) * (x1 - m11) + ((x1 - m11) * x12 + (x2 - m12) * x22) * (
            x2 - m12)) - 0.5 * (
                 ((x1 - m31) * z11 + (x2 - m32) * z21) * (x1 - m31) + ((x1 - m31) * z12 + (x2 - m32) * z22) * (
                     x2 - m32)) + 0.5 * np.log(np.linalg.det(SigmaX) / np.linalg.det(SigmaZ))

# Item 3

fig = plt.figure()
plt.plot(s1, s2, 'cx', label='Setosa')
plt.plot(y1, y2, 'k*', label='Versicilor')
plt.plot(z1, z2, 'g+', label='Virginica')
plt.contour(x1, x2, yz, 0, colors='red')
plt.contour(x1, x2, xy, 0, colors='blue')
plt.grid(True)
plt.title('Diskriminacione linije')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('2.3.png')

# Item 4

set1 = dataset.iloc[40:49, 0].values
set2 = dataset.iloc[40:49, 1].values
set3 = dataset.iloc[40:49, 2].values

ver1 = dataset.iloc[90:99, 0].values
ver2 = dataset.iloc[90:99, 1].values
ver3 = dataset.iloc[90:99, 2].values

vir1 = dataset.iloc[140:149, 0].values
vir2 = dataset.iloc[140:149, 1].values
vir3 = dataset.iloc[140:149, 2].values

x1p1 = set1
x2p1 = set2

h1 = 0.5 * (((x1p1 - m11) * x11 + (x2p1 - m12) * x21) * (x1p1 - m11) + ((x1p1 - m11) * x12 + (x2p1 - m12) * x22) * (
            x2p1 - m12)) - 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + (
            (x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (x2p1 - m22)) + 0.5 * np.log(
    np.linalg.det(SigmaX) / np.linalg.det(SigmaY))

greska1 = 0
for i in h1:
    if i > 0:
        greska1 = greska1 + 1

x1p1 = ver1
x2p1 = ver2

h2 = 0.5 * (((x1p1 - m11) * x11 + (x2p1 - m12) * x21) * (x1p1 - m11) + ((x1p1 - m11) * x12 + (x2p1 - m12) * x22) * (
            x2p1 - m12)) - 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + (
            (x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (x2p1 - m22)) + 0.5 * np.log(
    np.linalg.det(SigmaX) / np.linalg.det(SigmaY))

greska2 = 0
for i in h2:
    if i < 0:
        greska2 = greska2 + 1

Eps1 = ((greska1) / 9)
Eps2 = ((greska2) / 9)
P = 0.5 * Eps1 + 0.5 * Eps2
T = (1 - P) * 100
print(T)
print(greska1, greska2)

fig = plt.figure()
plt.plot(set1, set2, 'cx', label='Setosa')
plt.plot(ver1, ver2, 'k*', label='Versicolor')
plt.contour(x1, x2, xy, 0, colors='red')
plt.grid(True)
plt.title('Setosa i Versicolor klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('2.4.1.png')

x1p1 = ver1
x1p2 = ver2

h1 = 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + ((x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (
            x2p1 - m22)) - 0.5 * (((x1p1 - m31) * z11 + (x2p1 - m32) * z21) * (x1p1 - m31) + (
            (x1p1 - m31) * z12 + (x2p1 - m32) * z22) * (x2p1 - m32)) + 0.5 * np.log(
    np.linalg.det(SigmaY) / np.linalg.det(SigmaZ))

greska3 = 0
for i in h1:
    if i > 0:
        greska3 = greska3 + 1

x1p1 = vir1
x2p1 = vir2

h2 = 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + ((x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (
            x2p1 - m22)) - 0.5 * (((x1p1 - m31) * z11 + (x2p1 - m32) * z21) * (x1p1 - m31) + (
            (x1p1 - m31) * z12 + (x2p1 - m32) * z22) * (x2p1 - m32)) + 0.5 * np.log(
    np.linalg.det(SigmaY) / np.linalg.det(SigmaZ))

greska4 = 0
for i in h2:
    if i < 0:
        greska4 = greska4 + 1

Eps1 = ((greska3) / 9)
Eps2 = ((greska4) / 9)
P = 0.5 * Eps1 + 0.5 * Eps2
T = (1 - P) * 100
print(T)
print(greska3, greska4)

fig = plt.figure()
plt.plot(ver1, ver2, 'k*', label='Versicolor')
plt.plot(vir1, vir2, 'g+', label='Virginica')
plt.contour(x1, x2, yz, 0, colors='red')
plt.grid(True)
plt.title('Versicolor i Viriginica klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('2.4.2.png')

x1p1 = set1
x1p2 = set2

h1 = 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + ((x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (
            x2p1 - m22)) - 0.5 * (((x1p1 - m31) * z11 + (x2p1 - m32) * z21) * (x1p1 - m31) + (
            (x1p1 - m31) * z12 + (x2p1 - m32) * z22) * (x2p1 - m32)) + 0.5 * np.log(
    np.linalg.det(SigmaY) / np.linalg.det(SigmaZ))

greska5 = 0
for i in h1:
    if i > 0:
        greska5 = greska5 + 1

x1p1 = vir1
x2p1 = vir2

h2 = 0.5 * (((x1p1 - m21) * y11 + (x2p1 - m22) * y21) * (x1p1 - m21) + ((x1p1 - m21) * y12 + (x2p1 - m22) * y22) * (
            x2p1 - m22)) - 0.5 * (((x1p1 - m31) * z11 + (x2p1 - m32) * z21) * (x1p1 - m31) + (
            (x1p1 - m31) * z12 + (x2p1 - m32) * z22) * (x2p1 - m32)) + 0.5 * np.log(
    np.linalg.det(SigmaY) / np.linalg.det(SigmaZ))

greska6 = 0
for i in h2:
    if i < 0:
        greska6 = greska6 + 1

Eps1 = ((greska5) / 9)
Eps2 = ((greska6) / 9)
P = 0.5 * Eps1 + 0.5 * Eps2
T = (1 - P) * 100
print(T)
print(greska5, greska6)

fig = plt.figure()
plt.plot(set1, set2, 'cx', label='Setosa')
plt.plot(vir1, vir2, 'g+', label='Virginica')
plt.contour(x1, x2, xz, 0, colors='red')
plt.grid(True)
plt.title('Setosa i Viriginica klasifikator test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()
fig.savefig('2.4.3.png')
