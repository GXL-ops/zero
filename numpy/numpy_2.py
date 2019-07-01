import numpy as np

_file_path = "csv文件目录"

# delimiter：按照“，”分割, unpack=True表示转置。查看源码了解所有参数
np.loadtxt(_file_path, delimiter=",", dtype="int", unpack=True)

print("*" * 100)

# 转置
t1 = np.array(range(24)).reshape(4, 6)
t2 = t1.transpose()
t3 = t1.T
t4 = t1.swapaxes(1, 0)  # ------------交换轴
