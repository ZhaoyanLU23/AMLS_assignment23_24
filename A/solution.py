import numpy as np
import xgboost as xgb
import faulthandler

faulthandler.enable()


class Solution:
    def __init__(self):
        pass

    def solve(self, data_abs_path: str):
        print("=====================================")
        print("|         Solving Task A ...        |")
        print("=====================================")
        # TODO: add your solution here

        # Demo
        self.demo(data_abs_path)

        print("----------Task A finished!-----------")
        print()

    def demo(self, path: str):
        """Cost 2s on CPU.
        [[144  90]
        [  7 383]]
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

    def read(self, path: str) -> list:
        """Read data for task A.
        train_images (4708, 28, 28)
        val_images (524, 28, 28)
        test_images (624, 28, 28)
        train_labels (4708, 1)
        val_labels (524, 1)
        test_labels (624, 1)
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
