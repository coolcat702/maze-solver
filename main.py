"""
USAGE:
python main.py size heuristic display
python main.py excel start end sample
python main.py graph start end sample heuristic
"""
from math import sqrt
from queue import PriorityQueue
from sys import argv, setrecursionlimit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
setrecursionlimit(5000)

HEURISTICS = {
	'manhattan': lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]),
	'taxi': lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]),
	'grid': lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]),
	'dijkstra': lambda p1, p2: 0,
	'constant': lambda p1, p2: 0,
	'const': lambda p1, p2: 0,
	'chebyshev': lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])),
	'diagonal': lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])),
	'diag': lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])),
	'chebyshev': lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])),
	'diag-exact': lambda p1, p2: sqrt(2)*min(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) + abs(p1[0] - p2[0])+abs(p1[1] - p2[1]),
	'octile': lambda p1, p2: sqrt(2)*min(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) + abs(p1[0] - p2[0])+abs(p1[1] - p2[1]),
	'oct': lambda p1, p2: sqrt(2)*min(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) + abs(p1[0] - p2[0])+abs(p1[1] - p2[1]),
	'euclidean': lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2),
	'linear': lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2),
	'lin': lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
}

EMPTY = 1
SEARCHED = 0
PATH = 2
DIRECTIONS = {'R': (1, 0), 'D': (0, 1), 'L': (-1, 0), 'U': (0, -1)}

def isValidConnection(pos1, pos2, connections):
	return [pos1, pos2] in connections or [pos2, pos1] in connections

def generateMaze(size: int):
	maze = np.full((size, size), EMPTY)
	if size < 2:
		return maze, [[[]]]
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
	try:
		return HEURISTICS[heuristicFunc](point1, point2)
	except KeyError:
		print(f'invalid heuristic: {heuristicFunc}. options: manhattan (grid), constant (dijkstra), diagonal, euclidean, octile (diag-exact)')
		return -1

def solveMaze(maze, connections, heuristicFunc):
	size = maze.shape[0]
	if size < 2:
		return maze, 0, 1
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

def print_usage_error():

	print('usage: python main.py size heuristic printing (optional) or python main.py excel')

def plotVisual(argv):
	output = []
	if argv[2] == 'excel':
		for size in range(int(argv[2]), int(argv[3])+1):
			for heuristic in ['grid', 'const', 'diag', 'oct', 'lin']:
				total_searches = sum(solveMaze(*generateMaze(size), heuristic)[1] for _ in range(int(argv[4])))
				average_searches = total_searches / int(argv[4])
				output.append((size, heuristic, average_searches))
				print(f'{size} {heuristic}')
		df = pd.DataFrame(output, columns=['size', 'column', 'data'])
		df_pivot = df.pivot(index='size', columns='column', values='data')
		df_pivot.to_excel(f'output_AO{argv[4]}_{argv[2]}-{argv[3]}.xlsx', index=True)
	else:
		rowToPlot = argv[5]
		if rowToPlot != 'all':
			for size in range(int(argv[2]), int(argv[3])+1):
				maze, connections = generateMaze(size)
				total_searches = sum(solveMaze(maze, connections, rowToPlot)[1] for _ in range(int(argv[4])))
				average_searches = total_searches / int(argv[4])
				output.append((size, average_searches))
			df = pd.DataFrame(output, columns=['size', 'average searches'])
			df.plot(x='size', y='average searches', kind='line')
		else:
			output = []
			for heuristic in ['manhattan', 'constant', 'diagonal', 'octile', 'linear']:
				for size in range(int(argv[2]), int(argv[3])+1):
					maze, connections = generateMaze(size)
					total_searches = sum(solveMaze(maze, connections, heuristic)[1] for _ in range(int(argv[4])))
					average_searches = total_searches / int(argv[4])
					output.append((size, heuristic, average_searches))
			df = pd.DataFrame(output, columns=['size', 'heuristic', 'average searches'])
			df.pivot(index='size', columns='heuristic', values='average searches').plot(kind='line')
	plt.show()
def plotCMDLine(argv):
	size = int(argv[1])
	maze, connections = generateMaze(size)
	heuristicFunc = argv[2]
	start = time.time()
	solvedMaze, searches, length = solveMaze(np.copy(maze), connections, heuristicFunc)
	end = time.time()
	doPrint = argv[3] if len(argv) > 3 else 'all'
	if doPrint in ['unsolved', 'all']:
		print('Original Maze:')
		printMaze(maze, connections)
	if doPrint in ['solved', 'all']:
		print('\nSolved Maze:')
		printMaze(solvedMaze, connections)
	print(f'\nSolved in {end - start} seconds')
	print(f'found path {length} cells long')
	print(f'searched {searches} out of {size**2} cells\n')

def main():
	if len(argv) == 5 or len(argv) == 6 and argv[1] == 'excel' or argv[1] == 'graph':
		plotVisual(argv)
	elif len(argv) < 2 or len(argv) > 4:
		print_usage_error()
		return
	else:
		size = int(argv[1])
		plotCMDLine(argv)

if __name__ == '__main__':
	try:
		main()
	except RecursionError:
		print('error: mazeSize too large')
