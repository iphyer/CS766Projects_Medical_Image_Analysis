import matplotlib.pyplot as plt

loss = []
with open('loss.txt') as f:
    content = f.readlines()

for i in range(len(content)):
    loss.append(float(content[i].split()[-1]))

#print(loss)
plt.plot(loss[0:5000],linewidth=5,color = 'r')
plt.show()