import numpy as np
import xgboost as xgb


class Solution:
    def __init__(self):
        pass

    def solve(self, data_abs_path: str):
        print("=====================================")
        print("|         Solving Task B ...        |")
        print("=====================================")

        # TODO: Add your solution here
        self.demo(data_abs_path)

        print("----------Task B finished!-----------")
        print()

    def demo(self, path: str):
        """Cost one hour on CPU.
        [[1004    1    5    0   32  284    7    0    5]
        [   0  847    0    0    0    0    0    0    0]
        [   0    0  121    7    0  152    1   52    6]
        [   0    1   10  365  118    2   79    1   58]
        [  36   83    1   15  852    1   25    2   20]
        [   0    0  161    1    0  281   49   84   16]
        [   4    0    7   16   25   12  475    0  202]
        [   0    0  109    0    6   65   14  162   65]
        [   0    0   28   30   14   10  154    4  993]]
        """
        from sklearn.metrics import confusion_matrix

        X_train, X_test, y_train, y_test = self.read(path)
        print("Read data finished!")

        clf = xgb.XGBClassifier()
        print("Start fitting...")
        # Fit the model, test sets are used for early stopping.
        clf.fit(X_train, y_train)
        print("After fitting...")
        # Save model into JSON format.
        clf.save_model("clf.json")
        predictions = clf.predict(X_test)
        actuals = y_test
        print(confusion_matrix(actuals, predictions))
        print("Model saved!")

    def read(self, path: str):
        """
        train_images (89996, 28, 28, 3)
        val_images (10004, 28, 28, 3)
        test_images (7180, 28, 28, 3)
        train_labels (89996, 1)
        val_labels (10004, 1)
        test_labels (7180, 1)
        """

        data = np.load(path)

        # for k, v in data.items():
        #     print(k, v.shape)

        X_train = data.get("train_images", np.asarray([]))
        X_train = self._reshape_to_2_dims(X_train)
        # print(X_train.shape)
        # print(np.unique(X_train))
        y_train = data.get("train_labels", np.asarray([]))
        # print(y_train.shape)
        # print(np.unique(y_train))

        X_test = data.get("test_images", np.asarray([]))
        X_test = self._reshape_to_2_dims(X_test)
        # print(X_test.shape)
        # print(np.unique(X_test))
        y_test = data.get("test_labels", np.asarray([]))
        # print(y_test.shape)
        # print(np.unique(y_test))

        result = [
            X_train,
            X_test,
            y_train,
            y_test,
        ]
        return result

    def _reshape_to_2_dims(self, ndarray: np.ndarray) -> np.ndarray:
        from functools import reduce
        import operator

        n_samples = ndarray.shape[0]
        if n_samples > 0:
            n_features = reduce(operator.mul, ndarray.shape[1:])
            ndarray = ndarray.reshape((n_samples, n_features))
        return ndarray
