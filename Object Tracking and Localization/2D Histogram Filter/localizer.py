#import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []
    s = 0
    for row in grid:
        new_beliefs_row = []
        for item in row:
            hit = (color == item)
            new_beliefs_row.append((hit*p_hit) + (1-hit)*p_miss)
            s += (hit*p_hit) + (1-hit)*p_miss
        new_beliefs.append(new_beliefs_row)
    
    # divide all elements of q by the sum to normalize
    for i in range(len(new_beliefs)):
        for j in range(len(new_beliefs[i])):
            new_beliefs[i][j] = new_beliefs[i][j] / s
    #
    # TODO - implement this in part 2
    #
    
    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width  = len(beliefs[0])
    new_G  = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            #new_i = (i + dy ) % width
            #new_j = (j + dx ) % height
            new_i = (i + dy ) % height
            new_j = (j + dx ) % width
            #pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell  # this is the problem

    return blur(new_G, blurring)