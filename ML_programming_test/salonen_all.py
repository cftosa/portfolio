"""
Name: Tommi Salonen
Email: tommi.t.salonen@tuni.fi

DATA.ML.100 programming test 2
"""

import numpy as np
import matplotlib.pyplot as plt
import math

x_train = np.loadtxt("X_train.txt")
y_train = np.loadtxt("y_train.txt")
x_test = np.loadtxt("X_test.txt")
y_test = np.loadtxt("y_test.txt")

# a) Baseline classifier

male = 0
female = 1
correct_m = 0
correct_f = 0

for i in range(0, len(y_test)):
    if y_test[i] == male:
        correct_m += 1

for x in range(0, len(y_test)):
    if y_test[x] == female:
        correct_f += 1

print(f"Correct classification percentage for assuming all samples to be male: {correct_m / len(y_test) * 100:.2f}%")
print(f"Correct classification percentage for assuming all samples to be female: {correct_f / len(y_test) * 100:.2f}%")


# b) Gaussian pdf

male = []
female = []

for z in range(0, len(y_train)):
    if y_train[z] == 0:
        male.append(x_train[z][0])
    else:
        female.append(x_train[z][0])

male_array = np.array(male)
female_array = np.array(female)

mean_male = np.mean(male_array)
std_male = np.std(male_array)
mean_female = np.mean(female_array)
std_female = np.std(female_array)


def gaussian_pdf(value, mean, std):
    return 1/(std * math.sqrt(2 * math.pi)) * math.exp(-1/2 * (((value - mean)/std) ** 2))


male_gaussian = []
female_gaussian = []

for m in male_array:
    male_gaussian.append(gaussian_pdf(m, mean_male, std_male))

for f in female_array:
    female_gaussian.append(gaussian_pdf(f, mean_female, std_female))

male_gaussian = np.array(male_gaussian)
female_gaussian = np.array(female_gaussian)

fig, axs = plt.subplots()
plt.xlabel("Height in cm")
plt.ylabel("Computed likelihood")

axs.plot(male_array, male_gaussian, 'ko', label="Male")
axs.plot(female_array, female_gaussian, 'ro', label="Female")
axs.legend()

plt.show()


# c) Bayes classifier

test_samples = []

for k in range(0, len(y_test)):
    test_samples.append(x_test[k][0])

test_samples = np.array(test_samples)
correct = 0

for h in range(0, len(y_test)):
    probability_m = gaussian_pdf(test_samples[h], mean_male, std_male)
    probability_f = gaussian_pdf(test_samples[h], mean_female, std_female)

    if probability_m > probability_f:
        if y_test[h] == 0:
            correct += 1
    if probability_f > probability_m:
        if y_test[h] == 1:
            correct += 1

print(f"Correct classification percentage for the Bayes classifier: {correct / len(y_test) * 100:.2f}%")
