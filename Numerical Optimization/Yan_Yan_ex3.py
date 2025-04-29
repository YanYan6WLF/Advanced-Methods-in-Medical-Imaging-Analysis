import sympy as sp
from matplotlib import pyplot as plt
import numpy as np

# variables
X,Y=sp.symbols('X Y')

# function
f=3*((1-X)**2)*sp.exp((-X**2)-(Y+1)**2)\
    -10*(X/5-X**3-Y**5)*sp.exp(-X**2-Y**2)\
        -1/3*sp.exp(-(Y+1)**2-Y**2)
sp.pprint(f)

# partial derivative
f_x=sp.diff(f,X)
f_y=sp.diff(f,Y)

print('partial derivative of x: ', f_x)
print('partial derivative: ', f_y)


# vanilla gradient-descent algorithm
gamma=[0.1,0.01,0.001,0.0001]

# fig,axs=plt.subplots(4,1,figsize=(5,20))######


for r in range(4): # :
    g=gamma[r]
    x=1.0 ###
    y=0.5 ###
    x_val=[]
    y_val=[]
    F_val=[]
    T=[]
    t=0
    for i in range(100):
        f_x_val=f_x.evalf(subs={X:x,Y:y})
        f_y_val=f_y.evalf(subs={X:x,Y:y})
        f_val=f.evalf(subs={X:x,Y:y})
        

        x_val.append(x)
        y_val.append(y)
        F_val.append(f_val)
        T.append(t)

        x=x-g*f_x_val
        y=y-g*f_y_val
        t=t+1
    
    plt.plot(T,F_val,label=f'gamma={g}')
    plt.ylabel('Peaks function f(x, y) at each step')
    plt.xlabel('the number of steps')
    plt.title('Gradient Descent (Vanilla) of the Peaks function with start x=1.0, y=0.5')
    plt.legend()


plt.tight_layout()
plt.show()
plt.close()

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
x_vals=np.linspace(-3,3,100)###############
y_vals=np.linspace(-3,3,100)
xgrid,ygrid=np.meshgrid(x_vals,y_vals)#############
zgrid=sp.lambdify((X,Y),f,'numpy')(xgrid,ygrid)#################
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.set_title('Peaks Function in 3D')

surf=ax.plot_surface(xgrid,ygrid,zgrid,cmap='viridis')###########

fig.colorbar(surf)

plt.show()



        

