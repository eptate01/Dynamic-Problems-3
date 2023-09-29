import math
from copy import deepcopy

def BellmanFord(graph, start, weights):

    numVertices = len(graph[0])
    numEdges = sum([sum(row) for row in graph])

    dist = [[math.inf for _ in range(numVertices)] for _ in range(numEdges+1)]
    dist[0][start] = 0

    for i in range(1, numEdges+1):
        for z in range(numVertices):
            dist[i][z] = dist[i-1][z]
            for y in range(numVertices):
                if graph[y][z] and (dist[i][z] > dist[i-1][y] + weights[y][z]):
                    dist[i][z] = dist[i-1][y] + weights[y][z]

    negativeCycleVertices = set()
    negativeElements = math.inf

    while negativeElements != len(negativeCycleVertices):
        negativeElements = len(negativeCycleVertices)
        for first in range(numVertices):
            for second in range(first+1, numVertices):
                if(graph[first][second]):
                    if(dist[numEdges][second] + weights[first][second] < dist[numEdges][first] and dist[numEdges][first] != math.inf):
                        dist[numEdges][first] = dist[numEdges][second] + weights[first][second]
                        negativeCycleVertices.add(first)
                        negativeCycleVertices.add(second)
                        

    if(len(negativeCycleVertices) > 0):
        print("Negative edge cycle:")
        return sorted(negativeCycleVertices)

    return dist[numEdges]

def FloydWarshall(graph, weights):
    length = len(graph)
    negatives = False
    dist = [[math.inf for _ in range(len(graph))] for _ in range(len(graph))]
    for i in range(len(dist)):
        dist[i][i] = 0
    for i in range(length):
        for s in range(length):
            for t in range(length):
                if weights[i][t] != 0:
                    dist[s][t] = min(dist[s][t],dist[s][i] + weights[i][t])
    for i in range(length):
        if dist[i][i] < 0:
            negatives = True
    if negatives == True:
        print()
    else: 
        for i in range(length):
            for s in range(length):
                for t in range(length):
                    if weights[i][t] != 0:
                        dist[s][t] = min(dist[s][t],dist[s][i] + weights[i][t])
    
    return dist

def Problem3(matrix):

    dp = [[0 for col in matrix[0]] for row in matrix]
    dp[0][0] = matrix[0][0]

    for i in range(1, len(matrix)):
        dp[i][0] = dp[i-1][0] + matrix[i][0]

    for i in range(1, len(matrix[0])):
        dp[0][i] = dp[0][i-1] + matrix[0][i]

    for i in range(1, len(matrix)):
        for j in range(1, len(matrix[0])):    
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + matrix[i][j]

    return dp[-1][-1]

def Problem4(matrix):
    length = len(matrix)
    MaxSequenceIndex = []
    dp = [[-1 for i in range(length)] for i in range(length)]

    for i in range(length):
        for j in range(length):
            if dp[i][j] == -1:
                possMax = []
                dp[i][j] = Problem4_depthSearch(i,j,matrix,length, possMax)
                if len(possMax) > len(MaxSequenceIndex):
                    MaxSequenceIndex = deepcopy(possMax)
    
    MaxSequence = []
    for i in MaxSequenceIndex:
        MaxSequence.append(matrix[i[0]][i[1]])

    return MaxSequence

def Problem4_depthSearch(i,j,matrix,length, possMax):
    possMax.append([i,j])
    if i < 0 or j < 0:
        return
    if i >= length or j >= length:
        return
    if i < length-1 and matrix[i+1][j] == matrix[i][j]+1:
        Problem4_depthSearch(i+1, j, matrix, length, possMax)
    elif j < length-1 and matrix[i][j+1] == matrix[i][j] + 1:
        Problem4_depthSearch(i,j+1,matrix,length, possMax)
    elif i > 0 and matrix[i-1][j] == matrix[i][j]+ 1:
        Problem4_depthSearch(i-1,j,matrix,length, possMax)
    elif j > 0 and matrix[i][j-1] == matrix[i][j] + 1:
        Problem4_depthSearch(i,j-1,matrix,length, possMax)
    return 

        

    

if __name__ == '__main__':
    graph = [[0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]
    s = 0
    weight = [[0, 5, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 6, 4, 0],
        [0, 2, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]
    negWeight = [[0, 5, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, -6, 4, 0],
        [0, 2, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]
    print("BellmanFord no negative cycle: ", BellmanFord(graph, s, weight))
    print()
    print("BellmanFord with negative weight cycle: ")
    print(BellmanFord(graph, s, negWeight))
    print()

    print("Floyd Warshall no negative cycle: ")
    [print(i) for i in FloydWarshall(graph,weight)]
    print()
    print("Floyd Warshall with negative cycle: ")
    [print(i) for i in FloydWarshall(graph,negWeight)]


    m = [[4, 7, 8, 6, 4],
        [6, 7, 3, 9, 2],
        [3, 8, 1, 2, 4],
        [7, 1, 7, 3, 7],
        [2, 9, 8, 9, 3]]
    print()
    print(Problem3(m))


    matrix = [[1, 2, 9],
       [5, 3, 8],
       [4, 6, 7]]
    print()
    print(Problem4(matrix))
