import argparse
import numpy as np

import utils
import linear_model

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin, yBin)

        print("\nlogReg Training error %.3f" %
              utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" %
              utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin, yBin)

        print("\nlogRegL2 Training error %.3f" %
              utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" %
              utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # logRegL2 Training error 0.002
        # logRegL2 Validation error 0.074
        # nonZeros: 101

    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin, yBin)

        print("\nlogRegL1 Training error %.3f" %
              utils.classification_error(model.predict(XBin), yBin))
        print("logRegL1 Validation error %.3f" %
              utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # logRegL1 Training error 0.000
        # logRegL1 Validation error 0.052
        # nonZeros: 71

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin, yBin)

        print("\nTraining error %.3f" %
              utils.classification_error(model.predict(XBin), yBin))
        print("Validation error %.3f" % utils.classification_error(
            model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

        # Training error 0.000
        # Validation error 0.018
        # nonZeros: 24

    elif question == "2.5":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # TODO
        model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(XBin, yBin)

        print("\nTraining error %.3f" %
              utils.classification_error(model.predict(XBin), yBin))
        print("Validation error %.3f" % utils.classification_error(
            model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.coef_ != 0).sum())

        # Training error 0.002
        # Validation error 0.074
        # nonZeros: 101

        # Training error 0.000
        # Validation error 0.052
        # nonZeros: 71

    elif question == "3":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" %
              utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMulti)))

    elif question == "3.2":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" %
              utils.classification_error(model.predict(XMultiValid), yMultiValid))

        # logLinearClassifier Training error 0.084
        # logLinearClassifier Validation error 0.070

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(
            model.predict(XMultiValid), yMultiValid))

        # Training error 0.020
        # Validation error 0.036

    elif question == "3.5":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # TODO
        model = LogisticRegression(C=np.inf, fit_intercept=False)
        model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', C=np.inf, fit_intercept=False)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" %
              utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(
            model.predict(XMultiValid), yMultiValid))
