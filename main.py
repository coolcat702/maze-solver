"""
USAGE:
python main.py size heuristic sample_size
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

heuristics = {
	'manhattan': lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]),
	'constant': lambda p1, p2: 0,
	'chebyshev': lambda p1, p2: max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])),
	'octile': lambda p1, p2: sqrt(2)*min(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) + abs(p1[0] - p2[0])+abs(p1[1] - p2[1]),
	'euclidean': lambda p1, p2: sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2),
}

SEARCHED = 0
EMPTY = 1
PATH = 2
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def isValidConnection(pos1, pos2, connections):
	return [pos1, pos2] in connections or [pos2, pos1] in connections

def generate_maze(size: int):
	maze = np.full((size, size), EMPTY)
	if size < 2:
		return maze, [[[]]]
	connections = []

	def generateCell(x, y):
		visited[x, y] = True
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


def solve_maze(maze, connections, heuristic):
	size = maze.shape[0]
	if size < 2:
		return maze, 0, 1
	start = (0, 0)
	end = (size - 1, size - 1)
	if heuristics[heuristic](start, start) == -1:
		return 0, 0, 0
	open_queue = PriorityQueue()
	open_queue.put((heuristics[heuristic](start, end), 0, start))
	costs = {start: 0}
	visited = set()
	closed = {}
	searches = 1
	while not open_queue.empty():
		_, cost, current_node = open_queue.get()
		x, y = current_node

		if current_node == end:
			break

		if current_node in visited:
			continue

		visited.add(current_node)

		for direction in directions.values():
			next_node = (x + direction[0], y + direction[1])
			if not isValidConnection([x, y], list(next_node), connections):
				continue
			maze[next_node[0], next_node[1]] = SEARCHED
			new_cost = cost + 1
			if new_cost >= costs.get(next_node, float('inf')):
				continue

			costs[next_node] = new_cost
			closed[next_node] = current_node
			open_queue.put((new_cost + heuristics[heuristic](next_node, end), new_cost, next_node))
			searches += 1

	node = end
	path_length = 1
	while node != start:
		x, y = node
		maze[y, x] = PATH
		node = closed[node]
		path_length += 1
	maze[0, 0] = PATH
	return maze, searches, path_length

if __name__ == '__main__':
	if len(argv) != 4:
		print("Invalid arguments")
		return
	output = []
	start = int(argv[1])
	end = int(argv[2])
	sample = int(argv[3])
	for heuristic in ['manhattan', 'constant', 'diagonal', 'octile', 'linear']:
		for size in range(start, end+1):
			total_searches = 0
			for _ in range(sample):
				maze, connections = generate_maze(size)
				total_searches += solve_maze(maze, connections, heuristic)[1]
				average_searches = total_searches / sample
				output.append((size, heuristic, average_searches))
	df = pd.DataFrame(output, columns=['size', 'heuristic', 'average searches'])
	df.pivot(index='size', columns='heuristic', values='average searches').plot(kind='line')
	plt.show()
