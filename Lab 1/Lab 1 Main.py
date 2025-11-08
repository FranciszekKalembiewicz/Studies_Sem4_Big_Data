#2
for i in range(1,31):
    if i % 2 == 0:
        print(i)
    continue

#3
i = 0
x = 5
sum = 0
while i <= x:
    sum +=i
    i += 1
print(sum)

#4
def parzyste(x):
    sum = 0
    for i in range(0,x+1):
        if i % 2 == 0:
            sum +=i
    print(sum)
parzyste(5)

#5
for i in range(-100, 100):
    i *= -1
    if i % 2 == 0 and not i % 3 == 0 and not i % 8 == 0:
        print(i)

#6
n = 6
tab_all = []
temp_tab = []
for i in range(0, n):
    temp_tab = []
    for j in range(1, n+1):
        temp_tab.append(1)
    tab_all.append(temp_tab)

for i in range(1,n):
    for j in range(i,n):
        tab_all[i][j] = i+1
        tab_all[j][i] = i+1

for i in range(n):
    print(tab_all[i])
    print("\r")

#7
n = 5
list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O','P']
def devider(n,list):
    new_list = []
    for j in range(n):
        temp_list = []
        for i in range(j, len(list), n):
            temp_list.append(list[i])
        new_list.append(temp_list)
    return new_list

print(devider(n,list))

#8
list_main = [100, 90, 80, 70, 60, 50]
list_add = [49, 39, 29, 19]
list_main.pop(-1)
for i in range(0,len(list_add)):
    list_main.append(list_add[i])
print(list_main)

#9
list_main = ['A', 'B', 'C', 'D']
string = "Exit "

def function(list, string_add):
    for i in range(0,len(list)):
        list[i] = string_add + list[i]
    return list

print(function(list_main, string))

#10
list_main = [(1, 2, 3, 4), (4, 5, 6), (7, 8, 9)]
for i in range (0,len(list_main)):
    list_main[i] = list_main[i][:len(list_main[i])-1] + (0,)
print(list_main)

#11
list_main =  [(), (), ('',), ('i1', 'i2'), ('i1', 'i2', 'i3'), ('i4')]
wynik = []
for i in list_main:
    if not len(i) == 0:
        wynik.append(i)
print(wynik)

#12
dictionary = { 'f1': 4.8, 'f2': 2.4, 'f3': 1.2, 'f4': 0.6}
def dictionary_sum(dictionary):
    sum = 1
    for i in dictionary:
        sum *= dictionary[i]
print(dictionary_sum(dictionary))

#13
n = 6
dictionary = {}
for i in range(0,n+1):
    dictionary[i] = i**4
print(dictionary)

#14
dictionary = {1: 'A201', 2: 'B218', 3:'H018', 4:'B218', 5:'H018', 6: 'G123', 7: 'A007', 8: 'G230'}
def unique(dictionary):
    unique = []
    for i in dictionary.values():
        if i not in unique:
            unique.append(i)
    return unique
unique(dictionary)
print(unique(dictionary))