import numpy as np

EMPTY = 0
SEARCHED = 1
PATH = 2
WALL = 3

def generateMaze(size: int):
    path = np.array([[EMPTY for _ in range(size)] for _ in range(size)])
    return path

def solveWithDijkstra(maze: np.array):
    pass

def solveWithAStar(maze: np.array):
    pass

def printMaze(maze: np.array):
    for i in range(maze.shape[0]):
        line = ''
        for j in range(maze.shape[1]):
            cell_value = maze[i, j]
            if cell_value == EMPTY:
                line += ' '
            elif cell_value == SEARCHED:
                line += '/'
            elif cell_value == PATH:
                line += 'O'
            else:
                line += '#'
        print(f'| {line} |')

def main():
    printMaze(generateMaze(4))

if __name__ == '__main__':
    main()
