""" Batuhan Erden S004345 Department of Computer Science """


import itertools
import numpy as np


def solve(sudoku_array):
    solution, is_solved = solve_trying_best_3_brute_force(sudoku_array)

    if is_solved:
        return solution, True

    solution = insafe_solve(sudoku_array)
    return solution, False


def solve_trying_best_3_brute_force(sudoku_array):
    actual = sudoku_array[:, :, 0].copy()

    for i in range(len(sudoku_array)):
        for j in range(len(sudoku_array[0])):
            for k in range(len(sudoku_array[0][0])):
                actual[i, j] = int(sudoku_array[i, j, k])

                if actual[i, j] != 0:
                    try:
                        solver = SudokuSolver(actual)
                        solver.solve()

                        return solver.solution, True
                    except ValueError:
                        pass

            actual[i, j] = int(sudoku_array[i, j, 0])

    return sudoku_array[:, :, 0], False


class Clue:
    """
    A clue for solving the sudoku.
    Attributes:
        x              The X coordinate in a matrix of sudoku.
        y              The Y coordinate in a matrix of sudoku.
        possibilities  The list of possible values.
    """
    x = 0
    y = 0
    possibilities = []

    def __str__(self):
        return '(x=%d y=%d possibilities=%s)' % (self.x, self.y, self.possibilities)


class SudokuSolver:
    
    def __init__(self, sudoku, diagonal=False):
        sudoku = [[int(j) for j in i] for i in sudoku]
        
        self._n = len(sudoku)
        for row in sudoku:
            if len(row) != self._n:
                raise ValueError("The sudoku is missing some values.")
        
        # Basics.
        self._line = range(self._n)
        self._matrix = [[i // self._n, i % self._n] for i in range(self._n ** 2)]
        self._link_map = self._create_link_map(diagonal)

        # Depth matrix.
        self._depth_matrix = [[[float(len(self._link_map[i][j])), i, j] for j in self._line] for i in self._line]
        self._depth_line = list(itertools.chain.from_iterable(self._depth_matrix))
        # Calculate the current depth state. Initially, the ceil with most links is
        # the best choice to set into.
        k = max(e[0] for e in self._depth_line) + 2
        for e in self._depth_line:
            e[0] = self._n - e[0] / k

        # Superposition matrix.
        # noinspection PyUnusedLocal
        self._x = [[list(range(-self._n, 0)) for j in self._line] for i in self._line]
        # Apply the initial values.
        for i, j in self._matrix:
            value = sudoku[i][j]
            if value:
                self.set(value, i, j)

    def _create_link_map(self, diagonal=False):
        n_region = int(self._n ** .5)
        # Check for the correct input.
        if n_region ** 2 != self._n:
            raise ValueError("Unsupported size of sudoku.")
        region = [[i // n_region, i % n_region] for i in self._line]
        # Create mapping.
        m = []
        for i in self._line:
            column = []
            for j in self._line:
                ceil = []
                # Add row.
                ceil.extend([[e, j] for e in self._line if e != i])
                # Add column.
                ceil.extend([[i, e] for e in self._line if e != j])
                # Add region.
                for a, b in region:
                    x = a + i // n_region * n_region
                    y = b + j // n_region * n_region
                    if x != i and y != j:
                        ceil.append([x, y])
                if diagonal:
                    # Add main diagonal.
                    if i == j:
                        ceil.extend([[e, e] for e in self._line if e != i])
                    # Add sub-diagonal.
                    if i == self._n - j - 1:
                        ceil.extend([[e, self._n - e - 1] for e in self._line if e != j])
                column.append(ceil)
            m.append(column)
        return m

    def set(self, value, x, y):
        """
        :param value: The value to be set
        :param x: The X coordinate
        :param y: The Y coordinate
        """
        if 0 < value <= self._n and -value in self._x[x][y]:
            self._set(-value, x, y)
            self._depth_line.remove(self._depth_matrix[x][y])
        else:
            raise ValueError('Failed to set %d to [%d;%d]!' % (value, y + 1, x + 1))
        # Re-sort the depth map.
        self._depth_line.sort(key=lambda e: e[0])

    def clue(self, fast_search=True):
        """
        :return:
        The best possible step.
        """
        clue = Clue()
        clue.x = self._depth_line[0][1]
        clue.y = self._depth_line[0][2]
        clue.possibilities = [-e for e in self._x[clue.x][clue.y]]
        return clue

    def solve(self):
        """
        :return:
        <i>True</i> if one or more solutions of this sudoku exists,
        <i>False</i> otherwise.
        """
        solution = self._solve()
        self._x = solution
        return bool(solution)

    def _solve(self):
        if not self._depth_line:
            return self._x

        # Choose the best candidate.
        clue = self._depth_line[0]
        if not clue[0]:
            # Found an empty ceil with no
            # possible values.
            return None
        i, j = clue[1], clue[2]
        del self._depth_line[0]

        # Try all possibilities.
        x_value = self._x[i][j]
        for value in x_value:
            log = []
            self._set(value, i, j, log)
            self._depth_line.sort(key=lambda e: e[0])

            # Try to solve it.
            if self._solve() is not None:
                return self._x

            # Restore.
            for k in log:
                a, b = k >> 16, k & (1 << 16) - 1
                self._x[a][b].append(value)
                self._depth_matrix[a][b][0] += 1
        self._x[i][j] = x_value
        self._depth_line.insert(0, clue)
        self._depth_line.sort(key=lambda e: e[0])
        return None

    def _set(self, value, i, j, fallback=None):
        self._x[i][j] = [-value]

        # Remove this element from
        # other linked cells.
        for a, b in self._link_map[i][j]:
            try:
                self._x[a][b].remove(value)
                self._depth_matrix[a][b][0] -= 1
                # Remember the ceil's location
                if fallback is not None:
                    fallback.append(a << 16 | b)
            except ValueError:
                pass

    @property
    def solution(self):
        return np.array(self._x).reshape((9, 9))

    @staticmethod
    def format(x):
        return '\n'.join([' '.join([str(e[0]) for e in row]) for row in x])


def insafe_solve(sudoku_array):
    sudoku = sudoku_array[:, :, 0].copy()

    a = sudoku[0:3, 0:3]
    b = sudoku[0:3, 3:6]
    c = sudoku[0:3, 6:9]
    d = sudoku[3:6, 0:3]
    e = sudoku[3:6, 3:6]
    f = sudoku[3:6, 6:9]
    g = sudoku[6:9, 0:3]
    h = sudoku[6:9, 3:6]
    i = sudoku[6:9, 6:9]

    sub_sudokus = [a, b, c, d, e, f, g, h, i]

    for sub_sudoku in sub_sudokus:
        sub_sudoku[sub_sudoku == 0] = possibleEntry(sub_sudoku)
        crosshatch(sudoku)

    checksudoku(sudoku)
    return sudoku


def possibleEntry(arr2, arr1=np.arange(1, 10)):
    if len(arr1) == 0: # if comparing array is empty, just return the other array
        return arr2
    else:
        choice = np.setdiff1d(arr1, arr2)
        if len(choice) > 0: # if not empty join into a single number
            choice = int(''.join(str(i) for i in choice))
            return choice
        else: # if empty, return empty array
            return choice


def crosshatch(sudoku):
    # set array to add possible numbers to, based on row and column
    pool = np.empty(0, dtype=np.int64)

    sudoku_length = sudoku.shape[0]
    for x in range(sudoku_length):  # row
        for y in range(sudoku_length):  # column
            # if element is longer than 1 digit check its row and column
            if len(str(sudoku[x, y])) > 1:

                pool = np.append(pool, sudoku[x, :])  # append all the numbers in that element's row
                pool = np.append(pool, sudoku[:, y])  # append all the numbers in that element's column
                pool = pool[pool < 10]  # eliminate any number laster than 9

                # convert current element into an array of numbers
                current = np.array([int(i) for i in str(sudoku[x, y])], dtype=np.int64)

                # assign the element to a new possibleEntry
                try:
                    sudoku[x, y] = possibleEntry(pool, current)
                except ValueError:
                    sudoku[x, y] = 0

                # reset the pool for the next element
                pool = np.empty(0, dtype=np.int64)

    return sudoku


def checksudoku(sudoku):
    sudoku_length = sudoku.shape[0]
    for x in range(sudoku_length):  # row
        for y in range(sudoku_length):  # column
            if sudoku[x, y] > 9 or int(sudoku[x, y]) == 0:
                b = np.random.randint(3, 6, size=1)[0]
                sudoku[x, y] = (np.random.randint(1, 10, size=1)[0] ** b % 9) + 1

    return True

