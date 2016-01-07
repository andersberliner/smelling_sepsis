# curveshapes.py
# Anders Berliner
# 20160106

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(-10,30,0.1)
# plt.figure(figsize=(12,6))
# i = 0
# for A in [1]:
#     for B in [0, 1, 2]:
#         plt.subplot(3,1,i+1)
#         plt.title('A=%s,B=%s' % (A,B))
#         plt.xlabel('t')
#         plt.ylabel('sigmoids')
#         for k in [0.5, 1, 2]:
#                 y = 1/(A + np.exp(-k*t-B))
#                 label = 'B:%s,k:%s' % (B,k)
#                 plt.plot(t,y,label=label)
#         plt.axvline(-B, color='k')
#         plt.legend(fontsize=6)
#         i += 1
#         plt.ylim([0,1])
#
#
# plt.suptitle('Various 1/(A+exp(-kt-B)) curves')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # sigmoid, derivative of sigmoid, integral of sigmoid
# A, B, k = 1,0,0.5
#
# y = 1/(A + np.exp(-k*t+B))
# dy = k*np.exp(-k*t+B)/(A + np.exp(-k*t+B))**2
# iy = np.log(np.exp(B) + A*np.exp(k*t))
# # iy = np.log(A*np.exp(k*t))
#
#
# # plt.ylim([0,3])
#
#
# A2, B2, k2, C2 = 2,3,1, 1
# y2 = -C2/(A2 + np.exp(-k2*t+B2))
# dy2 = -C2*k2*np.exp(-k2*t+B2)/(A2 + np.exp(-k2*t+B2))**2
# # iy2 = -C2*np.log(np.exp(B2) + A2*np.exp(k2*t))
# iy2 = -C2*np.log(np.exp(B2) + A2*np.exp(k2*t))
#
# y3 = y + y2
# dy3 = dy + dy2
# iy3 = iy + iy2
#
# plt.subplot(3,1,1)
# plt.plot(t,y,'r', label='y')
# plt.plot(t,-y2,'g', label='y2')
# plt.plot(t,y3,'b', label='y-y2')
# plt.title('y')
# plt.legend()
#
# plt.subplot(3,1,2)
# plt.plot(t,dy,'r', label='dy')
# plt.plot(t,-dy2,'g', label='dy2')
# plt.plot(t,dy3,'b', label='dy-dy2')
# plt.legend()
# plt.title('dy')
#
# plt.subplot(3,1,3)
# plt.plot(t,iy,'r', label='iy')
# plt.plot(t,-iy2,'g',label='iy2')
# plt.plot(t,iy3,'b',label='iy-iy2')
# plt.title('iy')
# plt.legend()
#
# plt.suptitle('1/(1+exp(-0.5t+0))-1/(2+exp(t+3))')
#
# plt.show()
# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(t,-y2,'r', label='y2')
# # plt.plot(t,-y2**2,'r-', label='y2**2')
# plt.plot(t,-dy2,'g',label='dy2')
# mymax = (B2-np.log(A2))/k2
# plt.axvline(mymax, label='(B2-log(A2))/k2', color='k', linestyle='-')
# plt.axvline(B2, label='B2', color='purple', linestyle='-')
# plt.axhline(0.25, label='C2/A2/2', color='k')
# plt.axhline(0.5, label='C2/A2', color='orange', linestyle='-')
# plt.legend(fontsize=7)
# plt.xlim([-5,10])
# plt.ylim([0,0.55])
# plt.subplot(2,1,2)
# plt.plot(t,-iy2,'b', label='iy2')
# plt.plot(t,t,'k-.',label='y=t')
# plt.suptitle('C2/(A2+exp(-k2*t+B2) = 1/(2+exp(-t+3))')
# plt.legend()
# plt.ylim([0,10])
# plt.xlim([-5,10])
# plt.show()

def mylogit(x):
    return np.log(x/(1-x))

plt.figure(figsize=(12,6))
y = 1/(2+3*np.exp(-t+4))
ymax = 4-np.log(2/3.0)
y2 = 1/(1+3*np.exp(-t+4))
y2max = 4-np.log(1/3.0)
y3 = 1/(1+np.exp(-t+4))
y3max = 4-np.log(1/1.0)
y4 = 2*y

plt.subplot(3,1,1)
plt.plot(t,y, color='r', label='1/(2+3exp(-t+4))')
plt.axvline(ymax, color='r',linestyle=':', alpha=0.4)
plt.plot(t,y2,'b', label='1/(1+3exp(-t+4))')
plt.axvline(y2max, color='b',linestyle=':', alpha=0.4)
plt.plot(t,y3,'g', label='1/(1+exp(-t+4))')
plt.axvline(y3max,color='g',linestyle=':', alpha=0.4)
plt.plot(t,y4, 'k', label='2*y')
plt.legend(fontsize=9)

plt.subplot(3,1,2)
plt.plot(t,mylogit(y), color='r')
plt.plot(t,mylogit(y2), color='b')
plt.plot(t,mylogit(y3), color='g')
plt.plot(t,mylogit(y4), color='k')

plt.subplot(3,1,3)
plt.plot(t,y-ymax, color='r')
plt.plot()

plt.show()
