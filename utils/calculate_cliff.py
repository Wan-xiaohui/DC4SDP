def calculate_cliff(data1, data2):
    n = data1.shape[0]
    sum = 0
    for i in range(n):
        for j in range(n):
            if data1[i] > data2[j]:
                sum = sum + 1
            if data1[i] < data2[j]:
                sum = sum - 1
    return abs(sum)/(n*n)
