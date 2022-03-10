import numpy as np
import matplotlib.pyplot as plt
from potential import calc_phi

def gradient(f, X):
    """ 関数fの勾配を計算する。
    Parameters
    ----------
    X : ndarray
        座標。
    f : function
        関数。

    Returns
    -------
    grad : ndarray
        勾配。
    """
    h = 1e-4
    grad = np.zeros_like(X)

    for idx, x in enumerate(X):
        grad_tmp = np.zeros_like(x)

        for i in range(x.size):
            temp = x[i]
            x[i] = float(temp) + h
            fxh1 = f(x)

            x[i] = float(temp) - h
            fxh2 = f(x)

            grad_tmp[i] = (fxh1 - fxh2) / (2*h)
            x[i] = temp
        grad[idx] = grad_tmp
    return grad



if __name__ == "__main__":
    SCALE_SIZE = 50
    SCALE_STEP = 10

    # 座標を設定
    x = np.arange(-SCALE_SIZE, SCALE_SIZE, SCALE_STEP)
    y = np.arange(-SCALE_SIZE, SCALE_SIZE, SCALE_STEP)
    xx, yy = np.meshgrid(x, y)
    x0 = xx.flatten()
    x1 = yy.flatten()
    X = np.array([x0, x1]).T

    # 電場の計算
    ele_filed = - gradient(calc_phi, X)

    # グラフの描画
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111)

    ax.grid()
    ax.set_xlim(-SCALE_SIZE, SCALE_SIZE)
    ax.set_ylim(-SCALE_SIZE, SCALE_SIZE)

    ax.quiver(x0, x1, ele_filed.T[0], ele_filed.T[1], color = "red", angles = 'xy')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()