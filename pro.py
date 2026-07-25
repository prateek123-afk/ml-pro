import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
X=np.array([[30], [50], [44], [23], [37]])
y=np.array([0, 0, 1, 1, 1])
model=LogisticRegression()
model.fit(X,y)
x_values=np.linspace(10,100,100).reshape(-1,1)
y_prob=model.predict_proba(x_values)[:,1]
plt.scatter(X,y)
plt.plot(x_values,y_prob,color='red',marker='o',ms=20)
plt.xlabel('tempeature')
plt.ylabel('Probability of Rain')
plt.title("Logistic Regression Model-Weather Example")
plt.show()