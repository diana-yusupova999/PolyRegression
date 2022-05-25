# importing libraries
import numpy as np
import matplotlib.pyplot as plt
# importing libraries for polynomial transform
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline
# for calculating mean_squared error
from sklearn.metrics import mean_squared_error

def main():
    # creating a dataset with curvilinear relationship
    mas = [94, 102, 101, 105, 99, 105, 100, 100, 101, 100, 101, 99, 101, 111, 108, 110, 112, 113, 113, 116, 121, 120, 117, 119, 117, 115, 111, 107, 110, 108, 106, 101, 104, 103, 107, 110, 112, 117, 120, 128, 126, 131, 130, 130, 127, 127, 120, 111, 107, 97, 92, 92, 93]
    x = np.array([i for i in range(len(mas))])
    y = np.array(mas)
    coefficient = np.polyfit(x, y.reshape((-1, 1)), 9)
    coefficient2 = np.array2string(coefficient)
    f = open('koefreg.txt', 'a')
    f.write(coefficient2)
    f.close()
    # creating pipeline and fitting it on data
    Input = [('polynomial', PolynomialFeatures(degree=9)), ('modal', LinearRegression())]
    pipe = Pipeline(Input).fit(x.reshape(-1, 1), y)

    poly_pred = pipe.predict(x.reshape(-1, 1))
    # sorting predicted values with respect to predictor
    sorted_zip = sorted(zip(x, poly_pred))
    x_poly, poly_pred = zip(*sorted_zip)

    # plotting predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=15)
    plt.plot(x_poly, poly_pred, color='g', label='Polynomial Regression')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.legend()
    plt.show()

    print('RMSE for Polynomial Regression=>', np.sqrt(mean_squared_error(y, poly_pred)))
    r_sq = pipe.score(x.reshape((-1, 1)), y)
    print('coefficient of determination:', r_sq)

if __name__ == '__main__':
    main()
