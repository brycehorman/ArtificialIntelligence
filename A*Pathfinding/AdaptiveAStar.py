import numpy as np
import random as rr
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import shutil
import multiprocessing
import glob
import IPython
from state import State
import heapq


def randGridMaze(number, width=101, height=101):
    colorList = [(0,0,1),(1, 0, 0),(1,1,1),(0,0,0)]
    customMap= matplotlib.colors.ListedColormap(colorList)
    shape = (height,width)
    Z = np.random.choice([0,1], size=shape, p=[.70,.30])
    start = (0,0)
    end = (0,0)
    while(start == end):
        start = randomBlock(Z)
        end = randomBlock(Z)
    Z[start] = -1
    #the end node displayed in blue
    Z[end] = -2
    #print the before maze
    plt.figure()
    plt.imshow(Z, cmap=customMap, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.savefig("pics/randGrid/beforemaze{0:0=2d}.png".format(number))
    np.savetxt("arrs/randGrid/before{0:0=2d}.txt".format(number),Z,fmt='%d')
    Z[start] = 0
    Z[start] = 0
    #set the start and the end blocks to be red
    Z, adaptiveNum = adaptiveAStar(Z, start, end)
    repeatedNum = repeatedAStar(Z, start, end)[1]
    #create custom colormap to have blocked cells be black, unblocked be white, and those on the path be red
    
    plt.figure()
    plt.imshow(Z, cmap=customMap, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.savefig("pics/randGrid/maze{0:0=2d}.png".format(number))
    np.savetxt("arrs/randGrid/{0:0=2d}.txt".format(number),Z,fmt='%d')
    return (adaptiveNum, repeatedNum)

#method to choose a random position within the grid. Returns a tuple of x and y value
def randomBlock(z):
    x = rr.randint(0, z.shape[1]-1)
    y = rr.randint(0, z.shape[0]-1)
    while(z[x][y] != 0):
        x = rr.randint(0, z.shape[1]-1)
        y = rr.randint(0, z.shape[0]-1)
    return (x, y)


#driver method for a*, equivalent of the main method in his pseudocode
def adaptiveAStar(z, start, end):
    finalpath = [start]
    S = np.empty(z.shape, dtype=object)
    for i in range(z.shape[1]):
        for j in range(z.shape[0]):
            h = abs(i - end[0]) + abs(j - end[1])
            S[i][j] = State(h)
            S[i][j].pos = (i, j)
    counter = 0
    numExpanded = 0
    z2 = np.zeros_like(z)
    while(start != end):
        counter += 1
        S[start].setG(0)
        S[start].search = counter
        S[end].g = 9999
        S[end].search = counter
        o = []
        c = []
        S[start].pos = start
        heapq.heappush(o, (S[start].f, S[start].g, 0, S[start]))
        if(start[1] + 1 < z.shape[1]):
            z2[start[0],start[1]+1] = z[start[0],start[1]+1]
        if(start[1] -1 >= 0):
            z2[start[0],start[1]-1] = z[start[0],start[1]-1]
        if(start[0]+1 < z.shape[0]):
            z2[start[0]+1,start[1]] = z[start[0]+1,start[1]]
        if(start[0]-1 >=0):
            z2[start[0]-1,start[1]] = z[start[0]-1,start[1]]
        o, c, S, numExpanded = computePathLargestG(o, c, S, z2, start, end, counter, numExpanded)
        if(len(o) == 0):
            print('Target cannot be reached')
            return [z, numExpanded+1]
        initialpath = [end]
        curr = S[end]
        
        while(curr.pos != start):
            initialpath.append(curr.tree.pos)
            curr = curr.tree
        for x in c:
            newH = S[end].g - S[x.pos].g
            S[x.pos].h = newH
        for spot in reversed(initialpath):
            if(z[spot] == 1):
                break
            start = spot
            finalpath.append(spot)
    for spot in finalpath:
        z[spot] = -1
    z[end] = -2
    print('target reached')
    print('Adaptive Number of expanded nodes:', numExpanded)
    

    return [z, numExpanded]

def repeatedAStar(z, start, end):
    finalpath = [start]
    S = np.empty(z.shape, dtype=object)
    for i in range(z.shape[1]):
        for j in range(z.shape[0]):
            h = abs(i - end[0]) + abs(j - end[1])
            S[i][j] = State(h)
            S[i][j].pos = (i, j)
    counter = 0
    z2 = np.zeros_like(z)
    numExpanded = 0
    while(start != end):
        counter += 1
        S[start].setG(0)
        S[start].search = counter
        S[end].g = 9999
        S[end].search = counter
        o = []
        c = []
        heapq.heappush(o, (S[start].f, S[start].g, 0, S[start]))
        if(start[1] + 1 < z.shape[1]):
            z2[start[0],start[1]+1] = z[start[0],start[1]+1]
        if(start[1] -1 >= 0):
            z2[start[0],start[1]-1] = z[start[0],start[1]-1]
        if(start[0]+1 < z.shape[0]):
            z2[start[0]+1,start[1]] = z[start[0]+1,start[1]]
        if(start[0]-1 >=0):
            z2[start[0]-1,start[1]] = z[start[0]-1,start[1]]
        o, c, S, numExpanded = computePathLargestG(o, c, S, z2, start, end, counter, numExpanded)
        if(len(o) == 0):
            print('Target cannot be reached')
            return [z, numExpanded+1]
        initialpath = [end]
        curr = S[end]
        while(curr.pos != start):
            initialpath.append(curr.tree.pos)
            curr = curr.tree
        for spot in reversed(initialpath):
            if(z[spot] == 1):
                break
            start = spot
            finalpath.append(spot)
    for spot in finalpath:
        z[spot] = -1
    z[end] = -2
    print('target reached')
    print('Repeated Number of expanded nodes:', numExpanded)
    

    return [z, numExpanded]

        

#the actual a* process, this is repeated by my driver method
def computePathLargestG(o, c, S, z, start, end, counter, numExpanded):
    identifier = 1
    while(len(o) > 0 and S[end].g > o[0][0]):
        smallestF = heapq.heappop(o)
        numExpanded+=1
        smallestF = smallestF[3]
        if(isInOpen(smallestF, c)):
            continue
        c.append(smallestF)
        up = (smallestF.pos[0] -1 , smallestF.pos[1])
        right = (smallestF.pos[0], smallestF.pos[1]+1)
        down = (smallestF.pos[0] + 1, smallestF.pos[1])
        left = (smallestF.pos[0], smallestF.pos[1]-1)
        actions = [up, right, down, left]
        for a in actions:
            #check if the action is within the bounds of the grid and if it is unblocked
            if(a[0] >= 0 and a[0] < S.shape[0] and a[1] >= 0 and a[1] < S.shape[1] and z[a] != 1):
                if(S[a].search < counter):
                    S[a].setG(9999)
                    S[a].search = counter
                if(S[a].g > smallestF.g + 1):
                    S[a].setG(smallestF.g + 1)
                    S[a].tree = smallestF
                    if(isInOpen(S[a], o)):
                        continue
                    heapq.heappush(o, (S[a].f, -1*S[a].g, identifier, S[a]))
                    identifier = identifier + 1
    return [o, c, S, numExpanded]

def isInOpen(state, list):
    for x in list:
        if(x == state):
            return True
    return False


if __name__ == "__main__":
    if os.path.exists("arrs"):
        shutil.rmtree("arrs")
    if os.path.exists("pics"):
        shutil.rmtree("pics")
    if os.path.exists("maze.png"):
        os.remove("maze.png")
    
    for i in ["", "/randGrid/"]: 
        os.mkdir("pics"+i)
        os.mkdir("arrs"+i)

    ### specify the number of grids you want to generate
    n_grids = int(sys.argv[1])
    

    multiprocessing.freeze_support()
    num_proc = multiprocessing.cpu_count()
    ## for python 3.6 uncomment the line below, and comment the line above
    # num_proc = os.cpu_count()
    pool = multiprocessing.Pool(processes = num_proc)

    nn = [i for i in range(n_grids)]
    nn = pool.map(randGridMaze, nn)
    totalAdapt = 0
    totalRepeat = 0
    for x in nn:
        totalAdapt+=x[0]
        totalRepeat+=x[1]
    print("Average of adaptive a star:", totalAdapt/len(nn))
    print("Average of repeated a star:", totalRepeat/len(nn))



    pool.close()
    pool.join()