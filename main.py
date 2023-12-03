from collections import deque
from itertools import product
from queue import Queue
from os import path
from queue import PriorityQueue
from sqlite3 import connect
import numpy as np
import random
import sys
import time
sys.setrecursionlimit(1500)

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

def solveWithDijkstra(maze, connections):
    size = maze.shape[0]
    start = (0, 0)
    end = (size - 1, size - 1)
    queue = Queue()
    queue.put(start)
    costs = {start: 0}
    previous = {}
    while not queue.empty():
        currentNode = queue.get()
        x, y = currentNode

        if currentNode == end:
            break

        if maze[y, x] == SEARCHED:
            continue

        maze[y, x] = SEARCHED

        for direction in DIRECTIONS.values():
            next_node = (x + direction[0], y + direction[1])
            if not isValidConnection([x, y], list(next_node), connections):
                continue

            new_cost = costs[currentNode] + 1
            if next_node not in costs or new_cost < costs[next_node]:
                costs[next_node] = new_cost
                previous[next_node] = currentNode
                queue.put(next_node)

    node = end
    while node != start:
        x, y = node
        maze[y, x] = PATH
        node = previous[node]

    maze[0, 0] = PATH
    return maze

# HEURISTIC - manhattan distance for now, probably change later
def heuristic(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def solveWithAStar(maze, connections):

    size = maze.shape[0]
    start = (0, 0)
    end = (size - 1, size - 1)
    queue = PriorityQueue()
    queue.put((heuristic(start, end), 0, start))
    costs = {start: 0}
    visited = set()
    previous = {}

    while not queue.empty():
        _, cost, currentNode = queue.get()
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
            previous[next_node] = currentNode
            queue.put((new_cost + heuristic(next_node, end), new_cost, next_node))

    node = end
    while node != start:
        x, y = node
        maze[y, x] = PATH
        node = previous[node]

    maze[0, 0] = PATH
    return maze


def printMaze(maze: np.array, connections: np.array):
    val = np.full((2 * maze.shape[0] + 1, 2 * maze.shape[1] + 1), '#', order='C')

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
                updateConnection(y, x1, True, ':' if maze[y1, x1] == SEARCHED or maze[y2, x2] == SEARCHED else ' ')
        elif y1 == y2:
            x = min(x1, x2)
            updateConnection(y1, x, False, '-' if maze[y1, x1] == PATH and maze[y2, x2] == PATH else ':' if maze[y1, x1] == SEARCHED or maze[y2, x2] == SEARCHED else ' ')
    for row in val:
        print(''.join(row))

def main():
    size = int(sys.argv[1])
    if size > 55:
        raise ValueError("error: \'mazeSize\' too large")
    maze, connections = generateMaze(size)
    doPrint = sys.argv[3] if len(sys.argv) > 3 else 'all'
    if doPrint not in ['unsolved', 'solved', 'all']:
        raise ValueError("error: what to print unspecified")
    if doPrint in ['unsolved', 'all']:
        print("Original Maze:")
        printMaze(maze, connections)
    algorithm = sys.argv[2]
    if algorithm == 'astar':
        start = time.time()
        solvedMaze = solveWithAStar(np.copy(maze), connections)
        end = time.time()
    elif algorithm == 'dijkstra':
        start = time.time()
        solvedMaze = solveWithDijkstra(np.copy(maze), connections)
        end = time.time()
    else:
        raise ValueError('\'algorithm\' must be \'astar\' or \'dijkstra\'')
    if doPrint in ['solved', 'all']:
        print("\nSolved Maze:")
        printMaze(solvedMaze, connections)
    print(f'Solved in {end - start} seconds')

if __name__ == '__main__':
    main()
