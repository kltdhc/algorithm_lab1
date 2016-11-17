import csv
import math

storage = []
with open('../data/train.csv') as csvfile:
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

wfile = open('../data/data.txt.train', 'w')

print(category)

for line in storage[1:]:
    wfile.write('%s '%line[1])
    for j in range(2, len(storage[0])):
        if line[j][0] >= 'a' and line[j][0] <= 'z':
            temp = ord(line[j][0]) - ord('a')
            wfile.write('%d:1 '%(temp+axis[j]))
        else:
            wfile.write('%d:%s '%(axis[j], line[j]))
    wfile.write('\n')

wfile.close()
