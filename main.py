from queue import PriorityQueue
from sys import argv, setrecursionlimit, getrecursionlimit
from math import sqrt
import numpy as np
import random
import time

EMPTY = 1
SEARCHED = 0
PATH = 2
DIRECTIONS = {'R': (1, 0), 'D': (0, 1), 'L': (-1, 0), 'U': (0, -1)}

def isValidConnection(pos1, pos2, connections):
    return any(connection == [pos1, pos2] for connection in connections)

def generateMaze(size: int):
    maze = np.array([[EMPTY for i in range(size)] for j in range(size)])
    connections = []

    def generateCell(x, y):
        visited[x, y] = True

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)

        for xDistance, yDistance in directions:
            newX, newY = x + xDistance, y + yDistance

            if newX >= 0 and newX < size and newY >= 0 and newY < size and not visited[newX, newY]:
                connections.append([[x, y], [newX, newY]])
                generateCell(newX, newY)

    visited = np.zeros((size, size), dtype=bool)
    generateCell(0, 0)

    for connection in connections:
        x1, y1 = connection[0]
        x2, y2 = connection[1]
        maze[y1, x1] = 1
        maze[y2, x2] = 1

    return maze, connections

def heuristic(point1, point2, heuristicFunc):
    if heuristicFunc in ['manhattan', 'grid', 'taxi']:
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    elif heuristicFunc in ['dijkstra', 'constant', 'const']:
        return 0
    elif heuristicFunc in ['diag', 'diagonal', 'chebyshev']:
        return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))
    elif heuristicFunc in ['oct', 'octile', 'diag-exact']:
        return sqrt(2)*min(abs(point1[0] - point2[0]), abs(point1[1] - point2[1])) + abs(point1[0] - point2[0])+abs(point1[1] - point2[1])
    elif heuristicFunc in ['euclidean', 'euclid']:
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    else:
        print(f'invalid heuristic: {heuristicFunc}. options: manhattan (grid), constant (dijkstra), diagonal, euclidean, octile (diag-exact)')
        return -1

def solveMaze(maze, connections, heuristicFunc):
    size = maze.shape[0]
    start = (0, 0)
    end = (size - 1, size - 1)
    if heuristic(start, start, heuristicFunc) == -1:
        return 0, 0, 0
    openQueue = PriorityQueue()
    openQueue.put((heuristic(start, end, heuristicFunc), 0, start))
    costs = {start: 0}
    visited = set()
    closed = {}
    searches = 1
    while not openQueue.empty():
        _, cost, currentNode = openQueue.get()
        x, y = currentNode

        if currentNode == end:
            break

        if currentNode in visited:
            continue

        visited.add(currentNode)

        for direction in DIRECTIONS.values():
            next_node = (x + direction[0], y + direction[1])
            if not isValidConnection([x, y], list(next_node), connections):
                continue
            maze[next_node[0], next_node[1]] = SEARCHED
            new_cost = cost + 1
            if new_cost >= costs.get(next_node, float('inf')):
                continue

            costs[next_node] = new_cost
            closed[next_node] = currentNode
            openQueue.put((new_cost + heuristic(next_node, end, heuristicFunc), new_cost, next_node))
            searches += 1

    node = end
    pathLength = 1
    while node != start:
        x, y = node
        maze[y, x] = PATH
        node = closed[node]
        pathLength += 1
    maze[0, 0] = PATH
    return maze, searches, pathLength


def printMaze(maze: np.array, connections: np.array):
    val = np.full((2 * maze.shape[0] + 1, 2 * maze.shape[1] + 1), '@', order='C')

    def updateVal(i, j, char):
        val[i * 2 + 1, j * 2 + 1] = char

    def updateConnection(y, x, is_horizontal, char):
        i, j = (y * 2, x * 2 + 1) if is_horizontal else (y * 2 + 1, x * 2 + 2)
        val[i, j] = char

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            updateVal(i, j, ' ' if maze[i, j] == EMPTY else ':' if maze[i, j] == SEARCHED else '-')
    for (x1, y1), (x2, y2) in connections:
        if x1 == x2:
            y = min(y1, y2) + 1
            if maze[y1, x1] == PATH and maze[y2, x2] == PATH:
                updateConnection(y, x1, True, '|')
            else:
                updateConnection(y, x1, True, ' ' if maze[y1, x1] == SEARCHED or maze[y2, x2] == SEARCHED else ' ')
        elif y1 == y2:
            x = min(x1, x2)
            updateConnection(y1, x, False, '-' if maze[y1, x1] == PATH and maze[y2, x2] == PATH else ' ' if maze[y1, x1] == SEARCHED or maze[y2, x2] == SEARCHED else ' ')
    for row in val:
        print(''.join(row))

def main():

    if len(argv) < 3:
        print('usage: python main.py size heuristic printing (optional)')
        return None
    if len(argv) > 4:
        print('usage: python main.py size heuristic printing (optional)')
        return None

    size = int(argv[1])
    if size < 2:
        print(f'error: mazeSize {size} too small')
        return None
    try:
        maze, connections = generateMaze(size)
    except RecursionError:
        setrecursionlimit(getrecursionlimit() + 250)
        main()
        return None
    heuristicFunc = argv[2]
    start = time.time()
    solvedMaze, searches, length = solveMaze(np.copy(maze), connections, heuristicFunc)
    end = time.time()
    if isinstance(solvedMaze, int):
        print('usage: python main.py size heuristic printing (optional)')
        return None
    doPrint = argv[3] if len(argv) > 3 else 'all'
    if doPrint not in ['unsolved', 'solved', 'all', 'none']:
        print('error: what to print unspecified')
    if doPrint in ['unsolved', 'all']:
        print('Original Maze:')
        printMaze(maze, connections)
    if doPrint in ['solved', 'all']:
        print('\nSolved Maze:')
        printMaze(solvedMaze, connections)
    print(f'\nSolved in {end - start} seconds')
    print(f'found path {length} cells long')
    print(f'searched {searches} out of {size**2} cells\n')
if __name__ == '__main__':
    main()