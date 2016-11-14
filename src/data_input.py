import csv
import math

storage = []
with open('/home/wanghao/workspace/algorithm_lab1/train.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for line in reader:
        storage.append(line)

category = [1 for _ in range(0, len(storage[0]))]
for line in storage[1:]:
    for j in range(2, len(storage[0])):
        if line[j][0] >= 'a' and line[j][0] <= 'z':
            temp = ord(line[j][0]) - ord('a') + 1
            if temp > category[j]:
                category[j] = temp

axis = [0 for _ in range(0, len(storage[0]))]

for j in range(3, len(storage[0])):
    axis[j] = axis[j-1] + category[j-1]

wfile = open('/home/wanghao/workspace/algorithm_lab1/code/data/input_log.csv', 'w')

print(category)

for line in storage[1:]:
    wfile.write(line[1] + ' ')
    for j in range(2, len(storage[0])):
        if line[j][0] >= 'a' and line[j][0] <= 'z':
            temp = ord(line[j][0]) - ord('a')
            wfile.write('%d:1 '%(temp+axis[j]))
        else:
            wfile.write('%d:%.4f '%(axis[j], math.log(int(line[j])+1)))
    wfile.write('\n')

wfile.close()