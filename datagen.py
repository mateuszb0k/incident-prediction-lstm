import numpy as np
from matplotlib import pyplot as plt
data_dir = "dataset.csv"
np.random.seed(42)
def fun(x):
    return (np.sin(x) + np.cos(x))
samples = np.arange(0,50*np.pi,0.01)
# print(samples.shape)
data = fun(samples) + np.random.normal(loc=0,scale=0.1,size = samples.shape)
rand_samples = (np.random.choice(samples.size,size =100,replace=False)) #random samples without repeats
for s in rand_samples:
    for i in range(1,30):
        data[max(0,s-i)] += (30-i)/40
data[rand_samples] *=3.5
anomalies = np.zeros_like(data)
anomalies[rand_samples] = 1
data_to_save = np.column_stack((data,anomalies))
np.savetxt(data_dir,data_to_save,delimiter=",",fmt=["%.6f", "%d"])
print(np.mean(data))
print(np.std(data))
plt.plot(samples,data)
plt.show()