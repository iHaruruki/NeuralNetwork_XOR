import numpy as np

# ReLU関数
def relu_fanction(x):
    return np.where(x > 0, x, 0)

# ステップ関数
def step_function(x):
    return np.where(x > 0, 1, 0)

# XORゲートのニューラルネットワーク
def xor_gate(input1, input2):

    x = np.array([input1, input2])

    # 重みとバイアス
    W1 = np.array([[0.7, -0.8], [-0.4, 0.7]])   # 重み
    b1 = np.array([-0.2, -0.3])                 # バイアス

    # 中間層
    m1 = np.dot(x, W1) + b1
    print("中間層の出力: ", m1)

    # 中間層の出力をReLU関数に渡した結果
    z1 = relu_fanction(m1)

    # 出力層への重みとバイアス
    W2 = np.array([0.5, 0.6])
    b2 = np.array(-0.2)

    # 出力層
    out = np.dot(z1 * W2) + b2

    # 出力層の出力
    y = step_function(out)
    return y

    # 確認用コード
    print(xor_gate(0, 0))
    print(xor_gate(1, 0))
    print(xor_gate(0, 1))
    print(xor_gate(1, 1))
