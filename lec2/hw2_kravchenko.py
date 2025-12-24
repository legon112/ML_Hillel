# ДЗ №2 
# Author: Dmytro Kravchenko

import numpy as np

FILE_NAME = "lec2/iris.data.txt"

#1. Знайти в датасеті таргет та видалити цю колонку з датасету (видаляти за індексом)
data = np.genfromtxt(fname=FILE_NAME,
                  delimiter=',',
                  usecols=range(4), # останній стовпець - це таргет
                  filling_values=np.nan) # в дата сеті немає пропусків, але на всяк випадок

#2. Перетворити колонки, що залишились в 2D масив (або впевнитись, що це уже 2D масив)
if data.ndim != 2:
    print("ERROR: not 2D array")
    exit(1)

#3. Порахувати mean, median, standard deviation для 1-ї колонки
col1 = data[:, 0]
mean_col1 = np.mean(col1, axis=0)
median_col1 = np.median(col1, axis=0)
std_col1 = np.std(col1, axis=0)

print("Середнє 1-ї колонки:", mean_col1)
print("Медіана 1-ї колонки:", median_col1)
print("Стандартне відхилиння 1-ї колонки:", std_col1)

#4. Вставити 20 значень np.nan на випадкові позиції в масиві (при використанні звичайного рандому можуть накластись позиції, тому знайти рішення, яке гарантує 20 унікальних позицій)
# * Вирішив зробити через маску, щоб потренуватись її використовувати
mask = np.zeros(shape=data.size, dtype=bool) 
rand_ind = np.random.choice(mask.size, 20, replace=False) # replace гарантує унікальність
mask[rand_ind] = True
mask = mask.reshape(data.shape)
data[mask] = np.nan

#5. Знайти позиції вставлених значень np.nan в 1-й колонці
count_nan_col = np.isnan(data[:, 0]).sum()
print("Кількість nan в першій колонці:", count_nan_col)

#6. Відфільтрувати массив за умовою: значення в 3-й колонці > 1.5 та значения в 1-й колонці < 5.0 (зберегти у іншу змінну)
mask = np.logical_and(col1 < 5.0, data[:, 2] > 1.5)
filtered_data = data[mask]

#7. Замінити всі значення np.nan на 0
mask = np.isnan(data)
data[mask] = 0

#8. Порахувати всі унікальні значення в массиві та вивести їх разом із кількістю
uniqs, counts = np.unique(data, return_counts=True)
print("Унікальні значення та їх кількість:")
print(np.c_[uniqs, counts])

#9. Розбити масив по вертикалі на 2 рівні частини (не використовувати абсолютні числа, мають бути два массиви по 4 колонки)
data1, data2 = np.vsplit(data, 2)
print("Розміри розбитих масивів:")
print("data1 розмір:", data1.shape)
print("data2 розмір:", data2.shape)

#10. Відсортувати обидва массиви по 1-й колонці: 1-й за збільшенням, 2-й за зменшенням
arg_to_sort = data1[:, 0].argsort()
data1 = data1[arg_to_sort]
arg_to_sort = data2[:, 0].argsort()
data2 = data2[arg_to_sort[::-1]]

#11. Зібрати обидва массиви в одне ціле
data = np.concatenate([data1, data2], axis = 0)

#12. Знайти найбільш часто повторюване значення в массиві
num_of_max = uniqs[counts.argmax()]
print("Найбільш повторювальне число:", num_of_max)

#13. Написати функцію, яка б множила всі значення в колонці, які менше середнього значения в цій колонці, на 2, і ділила інші значення на 4.
def task_13(array: np.ndarray) -> np.ndarray:
    reshape_flag = False
    
    # Перевірка чи 1D масив, якщо так - робимо 2D для зручності
    if len(array.shape) != 2:
        array = array.reshape(-1, 1)
        reshape_flag = True
    
    # Обробка по колонках
    for col in range(array.shape[1]):
        mean = array[:,col].mean()
        mask = np.zeros(array.shape, dtype=bool)
        mask[:, col] = array[:, col] < mean
        array[mask] = array[mask] * 2
        mask[:,col] = np.logical_not(mask[:,col])
        array[mask] = array[mask] / 4
    
    # Повернення до 1D
    if reshape_flag:
        array = array.reshape(-1)
        
    return array

#14. Застосувати отриману функцію до 3-ї колонки
data[:,2] = task_13(data[:, 2])        