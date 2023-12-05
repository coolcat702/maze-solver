import pandas as pd
from sys import setrecursionlimit
from main import generateMaze, solveMaze
setrecursionlimit(2000)
output = []
for size in range(3, 31):
    for heuristic in ['grid', 'const', 'diag', 'oct', 'euclid']:
        res = 0
        for _ in range(10):
            maze, connections = generateMaze(size)
            _, searches, length = solveMaze(maze, connections, heuristic)
            res += searches
        res /= 10
        output.append((size, heuristic, res))
        print(f'{size} {heuristic}')
        df = pd.DataFrame(output, columns=['size', 'column', 'data'])
df_pivot = df.pivot(index='size', columns='column', values='data')
excel_file_path = 'output.xlsx'
df_pivot.to_excel(excel_file_path, index=True)
print('done')