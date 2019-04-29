import csv

with open('merged_result.csv', 'w') as f:
    writer = csv.writer(f)
    for i in range(1, 3):
        with open('result' + str(i) + '.csv', 'r') as file:
            reader = csv.reader(file)
            l = list(reader)
            for j in range(len(l)):
                l[j][0] = l[j][0].split('/')[-1]
                print(l[j])
                writer.writerow(l[j])
