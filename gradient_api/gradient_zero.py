# coding=utf-8
# author=uliontse

import re
import random
from typing import Union, Tuple, Callable

import numpy
import pandas
import sympy
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('Qt5Agg')
ArrayType = numpy.ndarray
DataFrameType = pandas.DataFrame


class LgzError(Exception):
    pass


class LimitOfGradientZero:
    def __init__(self):
        self.init_x = random.uniform(-10, 10)
        self.init_loss = float('inf')
        self.flag = 1

    @staticmethod
    def plt_plot_scatter(y_fn: Callable,
                         x: float,
                         plt_pause: float = 0.1,
                         plt_width: float = 10,
                         plt_grids: int = 200,
                         scatter_color: str = 'red'
                         ) -> None:
        curve_x_list = numpy.linspace(start=x-plt_width/2, stop=x+plt_width/2, num=plt_grids+1)
        plt.plot(curve_x_list, y_fn(curve_x_list))
        plt.scatter(x, y_fn(x), color=scatter_color)
        plt.pause(plt_pause)
        return

    @staticmethod
    def get_result(loops: int, loss: float, x: float, y: float, digit_precision: int) -> dict:
        result = {
            'loops': loops,
            'loss': round(loss, digit_precision),
            'x': round(x, digit_precision),
            'y': round(y, digit_precision),
        }
        return result

    @staticmethod
    def get_init_theta(method: str, length: int) -> ArrayType:
        init_theta = {
            'zeros': numpy.zeros(shape=(length,)),
            'ones': numpy.ones(shape=(length,)),
            'uniform': numpy.random.uniform(low=-1, high=1, size=(length,)),
            'random': numpy.random.random(size=(length,)),
            'normal': numpy.random.standard_normal(size=(length,)),
        }
        if method not in init_theta or length < 1:
            raise LgzError('Inputs Error')
        return init_theta[method]

    @staticmethod
    def get_gradient_from_expr(expr: str) -> Tuple:
        if not isinstance(expr, str):
            raise LgzError('Inputs Error')

        ranks = re.compile(r'x\*\*(\d+)').findall(expr)
        if not ranks or int(ranks[0]) != 2:
            raise LgzError('Inputs Error')

        y_fn = lambda x: eval(expr)
        x = sympy.symbols('x')
        gradient_symbol = sympy.diff(eval(expr), x)
        gradient_fn = lambda x: eval(str(gradient_symbol))
        return y_fn, gradient_fn

    @staticmethod
    def get_gradient_from_polys(polys: ArrayType) -> Tuple:
        if not isinstance(polys, ArrayType) or polys.ndim != 1 or polys.size != 3:
            raise LgzError('Inputs Error')

        y_fn = numpy.poly1d(polys)
        gradient_fn = numpy.poly1d.deriv(y_fn)
        return y_fn, gradient_fn

    @staticmethod
    def get_gradient_from_df(df: DataFrameType) -> Tuple:
        return ()

    def limit_of_quadratic(self,
                           inputs: Union[str, ArrayType],
                           num_loops: int = 1000,
                           lr: float = 1e-1,
                           max_loss: float = 1e-3,
                           digit_precision: int = 4,
                           if_print_result: bool = False,
                           is_detail_result: bool = False,
                           if_plot: bool = False,
                           plot_freq: int = 2,
                           plot_pause: float = 0.1,
                           plot_width: float = 10,
                           plot_grids: int = 200,
                           plot_scatter_color: str = 'red'
                           ) -> Union[float, dict]:

        cnt = 0
        cur_x = self.init_x
        prev_loss = cur_loss = self.init_loss
        y_fn, g_fn = self.get_gradient_from_expr(inputs) if isinstance(inputs, str) else self.get_gradient_from_polys(inputs)

        for i in range(num_loops):
            cnt += 1

            prev_x = cur_x
            cur_x -= lr * g_fn(prev_x) * self.flag
            cur_loss = abs(g_fn(cur_x))
            if cur_loss < max_loss:
                break
            if cur_loss > prev_loss:
                self.flag = -self.flag
            else:
                prev_loss = cur_loss

            if if_plot and i % plot_freq == 0:
                self.plt_plot_scatter(y_fn, cur_x, plot_pause, plot_width, plot_grids, plot_scatter_color)

            if if_print_result:
                print(self.get_result(cnt, cur_loss, cur_x, y_fn(cur_x), digit_precision))
        return self.get_result(cnt, cur_loss, cur_x, y_fn(cur_x), digit_precision) if is_detail_result else round(y_fn(cur_x), digit_precision)



    def limit_of_coefficient(self,
                             inputs_x: Union[DataFrameType, ArrayType],
                             inputs_y: Union[DataFrameType, ArrayType],
                             num_loops: int = 1000,
                             lr: float = 1e-3,
                             max_loss: float = 1e-3,
                             digit_precision: int = 4,
                             if_print_result: bool = False,
                             is_detail_result: bool = False,
                             ) -> Union[ArrayType, dict]:

        return dict()

# def generate_2d(dataFrame:pd.DataFrame,step=0.1,num_iters=1e4):
#     '''
#     :param dataFrame: pandas.DataFrame, like train_data with labels.
#     :param step: float,
#     :param num_iters: int or float,
#     :return: dict,
#     '''
#
#     arr = np.hstack([np.ones((len(dataFrame),1)),dataFrame])
#     X = pd.DataFrame(arr) # 列名未知
#     Y = X.pop(X.shape[1]-1)
#
#     init_theta = np.zeros(X.shape[1])
#     cur_theta = init_theta
#     num_iters = int(num_iters)
#
#     def y(theta):
#         try:
#             return np.sum((Y - X.dot(theta)) ** 2) / len(Y)
#         except:
#             return float('inf')
#
#     def gradient(theta):
#         try:
#             return X.T.dot(X.dot(theta) - Y) * 2 / len(X)
#         except:
#             return float('inf')
#
#
#     cur_num = 0
#     for i in range(num_iters):
#         cur_num += 1
#         prev_theta = cur_theta
#         cur_theta -= step * gradient(prev_theta)
#         if abs(y(cur_theta) - y(prev_theta)) < np.array([1e-4])[0]:
#             break
#
#     return {
#         'theta': cur_theta.values,
#         'Y': y(cur_theta),
#         'Gradient': gradient(cur_theta).values,
#         'Numloop': cur_num,
#         'Linear_Intercept': cur_theta[0],
#         'Linear_Coef': cur_theta[1:].values,
#     }


if __name__ == '__main__':
    lgz = LimitOfGradientZero()

    inputs1 = '-x**2-80*x+1'
    r1 = lgz.limit_of_quadratic(inputs1, lr=1e-1, if_print_result=True, is_detail_result=True, if_plot=True)
    print(r1)

    inputs2 = numpy.array([1, -2, 1])
    r2 = lgz.limit_of_quadratic(inputs2, lr=1e-1, if_print_result=True, is_detail_result=True, if_plot=True)
    print(r2)

