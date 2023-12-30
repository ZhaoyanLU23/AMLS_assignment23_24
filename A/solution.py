class Solution:
    def __init__(self):
        pass

    def solve(self):
        print("=====================================")
        print("|         Solving Task A ...        |")
        print("=====================================")
        # TODO: add your solution here

        print("----------Task A finished!-----------")
        print()

    def read(self):
        import numpy as np

        data = np.load("pneumoniamnist.npz")
        for k, v in data.items():
            print(k, v.shape)


"""
train_images (4708, 28, 28)
val_images (524, 28, 28)
test_images (624, 28, 28)
train_labels (4708, 1)
val_labels (524, 1)
test_labels (624, 1)
"""
