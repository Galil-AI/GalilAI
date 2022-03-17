import numpy as np
from os.path import exists



preds = np.zeros((9,9))
for i in range(9):
    for j in range(9):
        for k in range(10):
            if exists(f'SizeMass/Kmeans{i}_{j}_{k}.pickle'):
                with open(f'SizeDamping/Kmeans{i}_{j}_{k}.pickle', 'rb') as f:
                    pred, reward = dill.load(f)
                    preds[i][j] += pred

print("OOD detection result: \n", preds)

threshold = np.zeros((9,9))
for i in range(5):
    if exists(f'SizeMass/SA/kl_{i+5}.pickle'):
        with open(f'SizeMass/SA/kl_{i}.pickle', 'rb') as f:
            matrix = dill.load(f)
            threshold = np.add(threshold,matrix)
threshold = threshold/5

baseline = np.zeros((9,9))
for i in range(10):
    if exists(f'SizeMass/SA/kl_{i+5}.pickle'):
        with open(f'SizeMass/SA/kl_{i+5}.pickle', 'rb') as f:
            matrix = dill.load(f)
        for j in range(9):
            for k in range(9):
                if matrix[j][k]>threshold[j][k]:
                    baseline[j][k]+= 1

print("OOD detection baseline: \n", baseline)


# sns.heatmap(
#     preds,
#     annot=True,
#     cmap=colors,
#     xticklabels=dampings,
#     yticklabels=sizes,
#     vmin=0, vmax=10
# )
# plt.xlabel('Test Damping')
# plt.ylabel('Test Mass')
# plt.title('CausalWorld-trainSize-testDamping')

# plt.show()