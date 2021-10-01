import matplotlib.pyplot as plt
import numpy as np

from testFunctions import *
function = ackley


X=np.linspace(-6,6)
Y=np.linspace(-6,6)

x,y=np.meshgrid(X,Y)
vector = np.array([x, y])
f=function(vector)


fig =plt.figure(figsize=(9,9))
ax=plt.axes(projection='3d')
ax.contour3D(x,y,f,450)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f')
ax.set_title(function.__name__ + ' function')
ax.view_init(50,50)

plt.contour(x,y,f,15)
plt.show()