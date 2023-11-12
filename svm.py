import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import tkinter as tk

# Инициализация данных
data = {'X': [], 'Y': [], 'Class': []}

# Функция для добавления точек в списки данных
def add_point(event):
    x, y = event.xdata, event.ydata
    data['X'].append(x)
    data['Y'].append(y)
    data['Class'].append(1 if event.button == 1 else -1)

    color = 'lightgreen' if event.button == 1 else 'lightsalmon'
    plt.scatter(x, y, c=color, edgecolors='black', s=50)
    train_and_plot()  # Обновляем разделяющую прямую после добавления точки

# Функция для обучения SVM и рисования разделяющей прямой
def train_and_plot():
    if len(data['X']) < 2:
        print("Необходимо добавить минимум две точки для обучения.")
        return

    X = np.array(list(zip(data['X'], data['Y'])))
    y = np.array(data['Class'])

    # Обучение модели
    clf.fit(X, y)

    # Очищаем предыдущий график
    ax.clear()

    # Рисуем точки
    for i in range(len(data['X'])):
        color = 'lightgreen' if data['Class'][i] == 1 else 'lightsalmon'
        plt.scatter(data['X'][i], data['Y'][i], c=color, edgecolors='black', s=50)

    # Рисуем разделяющую прямую
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=[':'])
    plt.fill_between(xx.ravel(), yy.ravel(), Z, where=(Z > 0), interpolate=True, color='lightgreen', alpha=0.3)
    plt.fill_between(xx.ravel(), yy.ravel(), Z, where=(Z < 0), interpolate=True, color='lightsalmon', alpha=0.3)

    plt.draw()
    plt.show()

# Функция для добавления новой точки после обучения
def add_new_point(event):
    x, y = event.xdata, event.ydata
    new_point = np.array([[x, y]])
    prediction = clf.predict(new_point)

    color = 'lightgreen' if prediction == 1 else 'lightsalmon'
    plt.scatter(x, y, c=color, edgecolors='black', s=50)
    train_and_plot()  # Обновляем разделяющую прямую после добавления новой точки

# Создание окна tkinter
root = tk.Tk()
root.title("SVM Example")

# Создание поля для рисования
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title('Click to add points (left: class 1, right: class -1)')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# Привязка событий к функциям
fig.canvas.mpl_connect('button_press_event', add_point)
train_button = tk.Button(root, text="Train SVM", command=train_and_plot)
train_button.pack()
fig.canvas.mpl_connect('button_press_event', add_new_point)

# Создание SVM модели
clf = svm.SVC(kernel='linear', C=1.0)

# Запуск главного цикла tkinter
plt.show()
root.mainloop()
