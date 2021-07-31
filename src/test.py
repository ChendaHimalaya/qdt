import numpy as np

def func(a,b,c,d,e):
    return a+b+c+d+e



def wrapper(params,*args):
    return func(*params[:2],*args)


s=np.array([1,2,3,4,5])
print(func(*[],1,2,3,4,5))

