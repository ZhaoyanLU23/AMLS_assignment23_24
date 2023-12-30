class Solution:
    def __init__(self):
        pass

    def solve(self):
        print("=====================================")
        print("|         Solving Task B ...        |")
        print("=====================================")
        # TODO: Add your solution here
        print("----------Task B finished!-----------")
        print()

    def read(self):
        import numpy as np

        data = np.load("pneumoniamnist.npz")
        for k, v in data.items():
            print(k, v.shape)


"""
train_images (89996, 28, 28, 3)
val_images (10004, 28, 28, 3)
test_images (7180, 28, 28, 3)
train_labels (89996, 1)
val_labels (10004, 1)
test_labels (7180, 1)
"""
