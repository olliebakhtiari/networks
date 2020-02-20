class Matrix:
    def __init__(self, m, n):
        """ rows x columns. """
        self.m = m
        self.n = n
        self.data = [[0] * n for _ in range(m)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, item):
        self.data[idx] = item

    def __str__(self):
        return '\n'.join([' '.join([str(item) for item in row]) for row in self.data]) + '\n'

    def __eq__(self, mat):
        return mat.data == self.data

    def __add__(self, mat):
        if self.get_shape() != mat.get_shape():
            raise MatrixError('ranks not equal.')
        result = Matrix(self.m, self.n)
        for x in range(self.m):
            row = [sum(item) for item in zip(self.data[x], mat[x])]
            result[x] = row

        return result

    def __sub__(self, mat):
        if self.get_shape() != mat.get_shape():
            raise MatrixError('ranks not equal.')
        result = Matrix(self.m, self.n)
        for x in range(self.m):
            row = [item[0] - item[1] for item in zip(self.data[x], mat[x])]
            result[x] = row

        return result

    def __mul__(self, mat):
        mat_m, mat_n = mat.get_shape()
        if self.n != mat_m:
            raise MatrixError('Matrices cannot be multiplied.')
        mat_t = mat.getTranspose()
        mul_mat = Matrix(self.m, mat_n)
        for x in range(self.m):
            for y in range(mat_t.m):
                mul_mat[x][y] = sum([item[0] * item[1] for item in zip(self.data[x], mat_t[y])])

        return mul_mat

    @classmethod
    def make_matrix(cls, data):
        m = len(data)
        n = len(data[0])
        if any([len(row) != n for row in data[1:]]):
            raise MatrixError('inconsistent row length')
        mat = Matrix(m, n)
        mat.rows = data

        return mat

    @classmethod
    def from_list(cls, list_of_lists):
        data = list_of_lists[:]
        return cls.make_matrix(data)

    def get_shape(self):
        return self.m, self.n

    def transpose(self):
        self.m, self.n = self.n, self.m
        self.data = [list(item) for item in zip(*self.data)]

    def get_transpose(self):
        m, n = self.n, self.m
        transposed = Matrix(m, n)
        transposed.data = [list(item) for item in zip(*self.data)]

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
    mb = Matrix.from_list([[2, 2, 3], [2, 2, 3]])
    print(mb)
