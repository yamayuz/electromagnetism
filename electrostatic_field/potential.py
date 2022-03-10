import numpy as np
import matplotlib.pyplot as plt

def calc_phi(x, y, q=1):
    """ 真空中の点電荷に関する電位を計算する。
    Parameters
    ----------
    x,y : ndarray
        座標。
    q : int
        電荷。

    Returns
    -------
    phi : ndarray
        電位。
    """
    # 原点での発散を防ぐために分母にepsを足す。
    k = 9.0e+9
    eps = 1e-8
    r = np.sqrt(x**2 + y**2)
    return 2*k*q / (r + eps)



if __name__ == "__main__":
    # 座標を設定
    x = np.arange(-50, 50, 0.01)
    y = np.arange(-50, 50, 0.01)
    X, Y = np.meshgrid(x, y)
    
    # 各点における電位を計算
    z = calc_phi(X, Y)

    # グラフを描画
    plt.imshow(z, extent=[-50, 50, -50, 50], origin='lower',cmap='jet', vmin=0, vmax=4e+9)
    plt.colorbar ()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()