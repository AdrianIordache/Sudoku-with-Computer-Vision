import numpy as np
import pickle

# grid_dict = {'grid': grid}
# pickle_out = open("grid.pickle","wb")
# pickle.dump(grid_dict, pickle_out)
# pickle_out.close()


pickle_in = open("grid.pickle","rb")
grid_pickle = pickle.load(pickle_in)

grid = grid_pickle['grid']



color = 0
queue = []
colors = np.zeros((9, 9), dtype = np.uint8)
visited = np.zeros((9, 9), dtype = np.uint8)

di = [-1, 0, 1, 0]
dj = [0, 1, 0, -1]

for k in range(9):
    for t in range(9):
        if visited[k][t] == 0:
            i = k; j = t
            color += 1
            queue.append((i, j))

            while len(queue) != 0:
                (i, j) = queue.pop(0) 

                visited[i][j] = 1
                colors[i][j] = color

                if grid[2 * i - 1][2 * j] != 1 and i - 1 >= 0 and visited[i - 1][j] == 0:
                    queue.append((i - 1, j))

                if grid[2 * i][2 * j - 1] != 1 and j - 1 >= 0 and visited[i][j - 1] == 0:
                    queue.append((i, j - 1))

                if grid[2 * i][2 * j + 1] != 1 and j + 1 < 9 and visited[i][j + 1] == 0:
                    queue.append((i, j + 1))

                if grid[2 * i + 1][2 * j] != 1 and i + 1 < 9 and visited[i + 1][j] == 0:
                    queue.append((i + 1, j))


print(colors)