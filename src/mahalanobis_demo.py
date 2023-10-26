import numpy as np
import scipy.spatial

print("\nBegin Mahalanobis distance demo ")

data = np.array([[64.0, 580.0, 29.0],
                 [66.0, 570.0, 33.0],
                 [68.0, 590.0, 37.0],
                 [69.0, 660.0, 46.0],
                 [73.0, 600.0, 55.0]])
print("\nSource dataset: ")
print(data)

cm = np.cov(data, rowvar=False)
print("\nCovariance matrix: ")
print(cm)

np.set_printoptions(precision=4, suppress=True)
icm = np.linalg.inv(cm)
print("\nInverse covar matrix: ")
print(icm)

u = np.array([66.0, 570.0, 33.0])
v = np.array([69.0, 660.0, 46.0])
md = scipy.spatial.distance.mahalanobis(u, v, icm)
print("\nu = ", end=""); print(u)
print("v = ", end = ""); print(v)
print("\nMahalanobis distance(u,v) = %0.4f " % md)  # 2.5536

print("\nEnd demo ")