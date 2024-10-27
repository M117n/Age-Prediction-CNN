import numpy as np

def conv(matriz, kernel):
    convolution = np.zeros((matriz.shape[0] - kernel.shape[0] + 1, matriz.shape[1] - kernel.shape[1] + 1))

    for i in range(matriz.shape[0] - kernel.shape[0] + 1):
        for j in range(matriz.shape[1] - kernel.shape[1] + 1):
            convolution[i, j] = np.sum(matriz[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)

    return convolution

s = [[2, 1, 3], 
     [4, 0, 2], 
     [1, 5, 6]]

w = [[-1, -2], 
     [1,   2]]

print(conv(np.array(s), np.array(w)))