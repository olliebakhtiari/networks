import random


class Matrix:
    def __init__(self, m, n, zeroes=True):
        """ rows x columns. """
        if zeroes:
            self.data = [[0] * n for _ in range(m)]
        else:
            self.data = []
        self.m = m
        self.n = n

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __str__(self):
        return '\n'.join([' '.join([str(value) for value in row]) for row in self.data]) + '\n'

    def __eq__(self, mat):
        return mat.data == self.data

    def __add__(self, mat):
        if self.get_shape() != mat.get_shape():
            raise MatrixError('shapes not equivalent.')
        result = Matrix(self.m, self.n)
        for x in range(self.m):
            row = [sum(value) for value in zip(self.data[x], mat[x])]
            result[x] = row

        return result

    def __sub__(self, mat):
        if self.get_shape() != mat.get_shape():
            raise MatrixError('shapes not equivalent.')
        result = Matrix(self.m, self.n)
        for x in range(self.m):
            row = [value[0] - value[1] for value in zip(self.data[x], mat[x])]
            result[x] = row

        return result

    def __mul__(self, mat):
        mat_m, mat_n = mat.get_shape()
        if self.n != mat_m:
            raise MatrixError('matrices cannot be multiplied.')
        mat_t = mat.get_transpose()
        mul_mat = Matrix(self.m, mat_n)
        for x in range(self.m):
            for y in range(mat_t.m):
                mul_mat[x][y] = round(sum([value[0] * value[1] for value in zip(self.data[x], mat_t[y])]), 3)

        return mul_mat

    @classmethod
    def construct_matrix(cls, data):
        m = len(data)
        n = len(data[0])
        if any([len(row) != n for row in data[1:]]):
            raise MatrixError('inconsistent row length.')
        mat = Matrix(m, n, zeroes=False)
        mat.data = data

        return mat

    @classmethod
    def construct_matrix_from_list(cls, list_of_lists):
        return cls.construct_matrix(list_of_lists[:])

    @classmethod
    def construct_random_matrix(cls, m, n, low=0, high=10, d_type='int', precision=3):
        mat = Matrix(m, n, zeroes=False)
        if d_type == 'int':
            for x in range(m):
                mat.data.append([random.randrange(low, high) for _ in range(mat.n)])
        elif d_type == 'float':
            for x in range(m):
                mat.data.append([round(random.random(), precision) for _ in range(mat.n)])
        else:
            raise NotImplementedError('data type not supported.')

        return mat

    def get_shape(self):
        return self.m, self.n

    def transpose(self):
        self.m, self.n = self.n, self.m
        self.data = [list(value) for value in zip(*self.data)]

    def get_transpose(self):
        m, n = self.n, self.m
        transposed = Matrix(m, n)
        transposed.data = [list(value) for value in zip(*self.data)]

        return transposed

    def map(self, function):
        """ Apply function to every element in matrix. """
        for i in range(self.m):
            for j in range(self.n):
                value = self.data[i][j]
                self.data[i][j] = function(value, i, j)


class MatrixError(Exception):
    pass


if __name__ == '__main__':
    matrix = Matrix.construct_random_matrix(4, 4, d_type='float', precision=3)
    print(matrix)
    m2 = Matrix.construct_random_matrix(4, 4, d_type='float', precision=3)
    print(m2)
    print(matrix * m2)

