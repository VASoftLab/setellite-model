import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import sys

# Глобальные переменные
mu = 3.986004418e14 # Гравитационный параметр Земли (м^3/с^2)
R = 6378137 # Экваториальный радиус Земли (м)
J2 = 1.08263e-3 # Вторая зональная гармоника геопотенциала

# Параметры модели
m = 500 # Масса спутника (кг)
T = 5e-3 # Тяга двигателя (Н)

# Вспомогательные переменные
aT = T/m
c = 1.5*J2*mu*R**2

# Критическая дистанция (м)
critical_distance = 500000
# Параметр решателя (с)
max_step=10


def satellite_model(t, X, U=None):
    """
    Функция правых частей модели спутника
    
    Параметры:
    t: float - время
    X: array - вектор состояния [x, y, z, Vx, Vy, Vz]
    U: array - управление [ux, uy, uz] (нормированное)
    
    Возвращаемое значение:
    dX/dt: array - производная вектора состояния
    """

    # Распаковка вектора состояния
    x, y, z, Vx, Vy, Vz = X
    # Распаковка весов управляющего вектора
    if U is None:
        ux, uy, uz = 0, 0, 0
    ux, uy, uz = U
    
    r = np.sqrt(x*x+y*y+z*z) # Модуль радиус-вектора
    r2 = r**2
    r3 = r**3
    r5 = r**5

    if r < 1:
        return np.zeros(6)
    
    # Гравитационное ускорение
    ax_grav = -mu * x / r3
    ay_grav = -mu * y / r3
    az_grav = -mu * z / r3

    # Ускорение от второй зональной гармоники
    ax_j2 = c/r5 * x * (1 - 5*z*z/r2)
    ay_j2 = c/r5 * y * (1 - 5*z*z/r2)
    az_j2 = c/r5 * z * (3 - 5*z*z/r2)

    # Ускорение от тягового двигателя
    ax_thrust = aT * ux
    ay_thrust = aT * uy
    az_thrust = aT * uz
    
    # Суммарное ускорение
    ax = ax_grav + ax_j2 + ax_thrust
    ay = ay_grav + ay_j2 + ay_thrust
    az = az_grav + az_j2 + az_thrust

    return np.array([Vx, Vy, Vz, ax, ay, az])

def propagate_orbit(X0, t_span, close_approaches=None, U=None, method='DOP853', rtol=1e-10, atol=1e-13):
    """
    Функция расчета орбиты спутника

    Параметры:
    X0: array - вектор состояния [x, y, z, Vx, Vy, Vz]
    t_span: tuple - интервал времени моделирования (t_start, t_end)
    U: array - управление [ux, uy, uz] (нормированное)
    method: str - метод интегрирования (по умолчанию Рунге-Кутта 8-го порядка)
    rtol: float - относительная точность
    atol: float - абсолютная точность
    max_step: float - максимальный шаг интегрирования (сек)

    Возвращаемое значение:
    solution: OdeSolution - решение на интервале
    """

    def satellite_dynamics(t, X):
        if U is None or close_approaches is None:
            U_local = [0, 0, 0]
        else:
            U_local = U(t, close_approaches)
            # Нормировка управления
            norm_U = np.sqrt(U_local[0]**2 + U_local[1]**2 + U_local[2]**2)
            if norm_U > 0:
                U_local = [u / norm_U for u in U_local]
        
        return satellite_model(t, X, U_local)

    # Решение системы ДУ (расчет траектории)
    sol = solve_ivp(
        satellite_dynamics,
        t_span,
        X0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True  # Флаг разрешает интерполяцию в любой момент времени
    )

    return sol

def satellite_distance(pos_s1, pos_s2, critical_distance=None, t_span=None, print_info=False, file_name='approaches.txt'):
    """
    Функция вычисления евклидова расстояния между двумя спутниками
    
    Параметры:
    pos1: array - координаты первого спутника [x, y, z]
    pos2: array - координаты второго спутника [x, y, z]
    critical_distance: float - допустимое значение опасного сближения
    print_info: bool - флаг, отвечающий за вывод дополнительной информации
    
    Возвращает:
    tuple: (distance, close_approaches)
        distance: array - расстояние между объектами
        close_approaches: list - список словарей с параметрами опасных сближений
        Каждый словарь содержит:
            - 'start_idx': int - индекс начала сближения
            - 'end_idx': int - индекс конца сближения
            - 'start_time': float - время начала сближения
            - 'end_time': float - время конца сближения
            - 'min_distance': float - минимальное расстояние на интервале
            - 'min_distance_idx': int - индекс минимального расстояния
            - 'distances': array - массив расстояний на интервале
    """
    x_s1, y_s1, z_s1 = pos_s1
    x_s2, y_s2, z_s2 = pos_s2
    
    # TODO: Выполнить интерполяцию всей сетки измерений
    # Проверяем размерности массивов
    if len(x_s1) != len(x_s2):        
        # Обрезаем до минимальной длины
        min_len = min(len(x_s1), len(x_s2))
        x_s1 = x_s1[:min_len]
        y_s1 = y_s1[:min_len]
        z_s1 = z_s1[:min_len]
        x_s2 = x_s2[:min_len]
        y_s2 = y_s2[:min_len]
        z_s2 = z_s2[:min_len]
        if t_span is not None:
            t_span = t_span[:min_len]
    
    distance = np.sqrt((x_s2 - x_s1)**2 + (y_s2 - y_s1)**2 + (z_s2 - z_s1)**2)
    
    # Словарь опасных сближений
    dangerous_approaches = []
    
    # Если задан порог, ищем опасные сближения
    if critical_distance is not None and t_span is not None:
        # Находим индексы, где расстояние меньше порога
        dangerous = distance < critical_distance
        
        if np.any(dangerous):
            # Находим границы интервалов опасного сближения
            diff = np.diff(np.concatenate(([0], dangerous.astype(int), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Для каждого интервала собираем параметры
            for start_idx, end_idx in zip(starts, ends):
                
                # Пропускаем интервал в начальный момент времени
                # (момент вывода спутников на орбиту)
                if (start_idx == 0):
                    continue
                
                # Находим минимальное расстояние на интервале
                interval_distances = distance[start_idx:end_idx]
                min_dist_idx_rel = np.argmin(interval_distances)
                min_dist_idx = start_idx + min_dist_idx_rel
                
                dangerous_approaches.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': t_span[start_idx], # Перевод в минуты
                    'end_time': t_span[end_idx], # Перевод в минуты
                    'min_distance': interval_distances[min_dist_idx_rel],
                    'min_distance_idx': min_dist_idx,
                    'distances': interval_distances.copy()
                })
            
            if print_info:
                # Выводим информацию о найденных опасных сближениях
                print(f'Найдено опасных сближений: {len(dangerous_approaches)}')
                print('##################################################')                
                for i, approach in enumerate(dangerous_approaches):
                    print(f'Сближение {i+1}:')
                    print(f'\tМинимальное расстояние: {approach["min_distance"]:.2f} (км)')
                    print(f'\tИнтервал: ({approach["start_time"] / 60:.2f} - {approach["end_time"] / 60:.2f}) (мин)')
                print('##################################################')
                # Сохраняем в файл
                save_approaches_to_file(dangerous_approaches, file_name)
                
    
    return distance, dangerous_approaches

def save_approaches_to_file(approaches, filename):
    """Построчное сохранение в текстовый файл"""
    # Получаем путь к директории текущего скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Создаем полный путь к файлу
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as file:
        for i, approach in enumerate(approaches, 1):
            file.write(f"Сближение #{i}\n")
            file.write(f"start_idx: {approach['start_idx']}\n")
            file.write(f"end_idx: {approach['end_idx']}\n")
            file.write(f"start_time: {approach['start_time'] / 60:.2f}\n")
            file.write(f"end_time: {approach['end_time'] / 60:.2f}\n")
            file.write(f"min_distance: {approach['min_distance']:.2f}\n")
            file.write(f"min_distance_idx: {approach['min_distance_idx']:.2f}\n")
            file.write(f"distances: {approach['distances']}\n")
            file.write("-" * 40 + "\n\n")
            
def collision_avoidance_controller(t, close_approaches):
    """
    Функция управления для предотвращения столкновений
    Начинает маневр за X минут до опасного сближения
    """
    # Время упреждения - 30 минут (1800 секунд)
    lead_time = 1800
    
    filename='controller.txt'
    # Получаем путь к директории текущего скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Создаем полный путь к файлу
    filepath = os.path.join(script_dir, filename)
    
    danger_time_start = 0.0
    danger_time_end = 0.0
    
    # Если нет опасных сближений, возвращаем нулевое управление
    if not close_approaches:
        #with open(filepath, 'a', encoding='utf-8') as file:
        #    file.write(f"{t:.2f}\t{danger_time_start:.2f}-{t:.2f}-{danger_time_end:.2f}\t[0.0, 0.0, 0.0]\n")
        return [0.0, 0.0, 0.0]
    
    # Коэффициенты управления
    u_x = 0
    u_y = 1
    u_z = 0
    
    # Проверяем каждое опасное сближение в базе
    for approach in close_approaches:        
        # Получаем интервал времени опасного сближения
        if 'start_time' in approach and 'end_time' in approach:
            danger_time_start = approach.get('start_time', float('inf'))
            danger_time_end = approach.get('end_time', float('inf'))            
            
            #####################################################################
            # Попали в участок сближения
            #####################################################################
            if danger_time_start - lead_time <= t <= danger_time_end + lead_time:
                # Начало уклонения
                if danger_time_start - lead_time <= t <= danger_time_start:
                    #with open(filepath, 'a', encoding='utf-8') as file:
                    #    file.write(f"{t:.2f}\t{danger_time_start:.2f}-{t:.2f}-{danger_time_end:.2f}\t[{u_x}, {u_y}, {u_z}]\tсход с орбиты\n")
                    return [u_x, u_y, u_z]
                
                # Проходим опасный участок
                if danger_time_start < t < danger_time_end:
                    #with open(filepath, 'a', encoding='utf-8') as file:
                    #    file.write(f"{t:.2f}\t{danger_time_start:.2f}-{t:.2f}-{danger_time_end:.2f}\t[0.0, 0.0, 0.0]\tпроход опасного участка\n")
                    return [0.0, 0.0, 0.0]
                
                # Возврат на орбиту
                if danger_time_end <= t <=  danger_time_end + lead_time:
                    #with open(filepath, 'a', encoding='utf-8') as file:
                    #    file.write(f"{t:.2f}\t{danger_time_start:.2f}-{t:.2f}-{danger_time_end:.2f}\t[{-u_x}, {-u_y}, {-u_z}]\tвозврат на орбиту\n")                
                    return [-u_x, -u_y, -u_z]            
                
                break
            #####################################################################
    
    danger_time_start = 0.0
    danger_time_end = 0.0
    # Вектор управления без коррекции
    #with open(filepath, 'a', encoding='utf-8') as file:
    #        file.write(f"{t:.2f}\t{danger_time_start:.2f}-{t:.2f}-{danger_time_end:.2f}\t[0.0, 0.0, 0.0]\n")
    return [0, 0, 0]

def plot_orbit_3d(sol_s1, sol_s2, show_earth=True):
    """
    Построение 3D графика орбиты
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Поворачиваем оси
    ax.view_init(elev=45, azim=-135)
    
    # Координаты центра масс 1-го спутника
    x_s1 = sol_s1.y[0] / 1000  # Переводим в км.
    y_s1 = sol_s1.y[1] / 1000
    z_s1 = sol_s1.y[2] / 1000
    
    # Координаты центра масс 2-го спутника
    x_s2 = sol_s2.y[0] / 1000  # Переводим в км.
    y_s2 = sol_s2.y[1] / 1000
    z_s2 = sol_s2.y[2] / 1000
    
    # Расстояния между спутниками    
    distance, close_approaches = satellite_distance(
        [x_s1, y_s1, z_s1],
        [x_s2, y_s2, z_s2],        
        critical_distance / 1000,
        sol_s1.t,
        False
        )
    
    # Индексы критических состояний
    critical_indexses = distance <= critical_distance / 1000
    
    if len(x_s1) != len(distance):        
        # Обрезаем до минимальной длины
        min_len = min(len(x_s1), len(x_s2), len(distance))
        x_s1 = x_s1[:min_len]
        y_s1 = y_s1[:min_len]
        z_s1 = z_s1[:min_len]
        x_s2 = x_s2[:min_len]
        y_s2 = y_s2[:min_len]
        z_s2 = z_s2[:min_len]        
    
    # Строим орбиту 1-го спутника
    ax.plot(x_s1, y_s1, z_s1, 'b-', linewidth=2, label='Орбита 1-го спутника')
    # Строим орбиту 2-го спутника
    ax.plot(x_s2, y_s2, z_s2, 'g-', linewidth=2, label='Орбита 2-го спутника')
    
    if (len(critical_indexses) > 0):
        # Критические координаты первого спутника
        x_s1_critical = x_s1[critical_indexses]
        y_s1_critical = y_s1[critical_indexses]
        z_s1_critical = z_s1[critical_indexses]
        
        x_s2_critical = x_s2[critical_indexses]
        y_s2_critical = y_s2[critical_indexses]
        z_s2_critical = z_s2[critical_indexses]
        
        # Отмечаем критические точки 1-го спутника
        ax.scatter([x_s1_critical], [y_s1_critical], [z_s1_critical], color='red', s=50, marker='o', alpha=0.1)
        # Отмечаем критические точки 2-го спутника
        ax.scatter([x_s2_critical], [y_s2_critical], [z_s2_critical], color='red', s=50, marker='o', alpha=0.1)    
    
    # Добавляем Землю (сфера)
    if show_earth:
        # Создаем сферу для Земли
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = (R / 1000) * np.outer(np.cos(u), np.sin(v))
        y_earth = (R / 1000) * np.outer(np.sin(u), np.sin(v))
        z_earth = (R / 1000) * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.15)
    
    # Настройка графика
    max_range = max(np.max(x_s1) - np.min(x_s1), 
                   np.max(y_s1) - np.min(y_s1), 
                   np.max(z_s1) - np.min(z_s1)) / 2
    mid_x = (np.max(x_s1) + np.min(x_s1)) / 2
    mid_y = (np.max(y_s1) + np.min(y_s1)) / 2
    mid_z = (np.max(z_s1) + np.min(z_s1)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (км)')
    ax.set_ylabel('Y (км)')
    ax.set_zlabel('Z (км)')
    ax.set_title('3D траектория спутника')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_orbit_projections(sol):
    """
    Построение проекций орбиты на координатные плоскости
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Извлекаем координаты
    x = sol.y[0] / 1000  # (км)
    y = sol.y[1] / 1000
    z = sol.y[2] / 1000
    t = sol.t / 60  # время в минутах
    
    # Проекция на XY
    ax = axes[0, 0]
    ax.plot(x, y, 'b-', linewidth=1.5)
    
    # Добавляем окружность Земли
    theta = np.linspace(0, 2*np.pi, 100)
    x_earth = (R / 1000) * np.cos(theta)
    y_earth = (R / 1000) * np.sin(theta)
    ax.plot(x_earth, y_earth, 'b--', alpha=0.5, label='Земля')
    
    ax.set_xlabel('X (км)')
    ax.set_ylabel('Y (км)')
    ax.set_title('Проекция на плоскость XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Проекция на XZ
    ax = axes[0, 1]
    ax.plot(x, z, 'b-', linewidth=1.5)
    
    # Добавляем Землю
    x_earth = (R / 1000) * np.cos(theta)
    z_earth = (R / 1000) * np.sin(theta)
    ax.plot(x_earth, z_earth, 'b--', alpha=0.5)
    
    ax.set_xlabel('X (км)')
    ax.set_ylabel('Z (км)')
    ax.set_title('Проекция на плоскость XZ')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Проекция на YZ
    ax = axes[1, 0]
    ax.plot(y, z, 'b-', linewidth=1.5)
    
    # Добавляем Землю
    ax.plot(y_earth, z_earth, 'b--', alpha=0.5)
    
    ax.set_xlabel('Y (км)')
    ax.set_ylabel('Z (км)')
    ax.set_title('Проекция на плоскость YZ')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Высота орбиты как функция времени
    ax = axes[1, 1]
    r = np.sqrt(x**2 + y**2 + z**2)  # Радиус (км)
    height = r - R/1000  # Высота над поверхностью (км)
    
    ax.plot(t, height, 'b-', linewidth=2)
    ax.axhline(y=800, color='r', linestyle='--', label='Опорная высота')
    ax.set_xlabel('Время (мин)')
    ax.set_ylabel('Высота (км)')
    ax.set_title('Высота орбиты')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Проекции орбиты спутника', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_distance(t_span, distance, threshold=None):
    """
    Функция отрисовки расстояния между спутниками
    
    Параметры:
    t_span: array - массив временных отсчетов
    distance: array - массив расстояний
    threshold: float - пороговое значение (опасная дистанция)
    """
    
    # TODO: Выполнить интерполяцию всей сетки измерений
    # Проверяем размерности массивов
    if len(t_span) != len(distance):        
        # Обрезаем до минимальной длины
        min_len = min(len(t_span), len(distance))
        t_span = t_span[:min_len]
        distance = distance[:min_len]
    
    # Создание фигуры
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Переводим в минуты
    t = t_span / 60
    
    # Основной график
    ax.plot(t, distance, 'b-', linewidth=1.5, label='Расстояние')
    
    # Опасная зона
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--',  linewidth=1, label=f'Порог: {threshold} (км)')
        
        dangerous = distance < threshold
        if np.any(dangerous):
            ax.fill_between(t, 0, distance, where=dangerous, color='red', alpha=0.2)
    
    # Настройки
    ax.set_xlabel('Время (м)')
    ax.set_ylabel('Расстояние (км)')
    ax.set_title('Расстояние между спутниками')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()    
    
    return fig


if __name__ == "__main__":
    
    # Удаление лога управления
    filename='controller.txt'    
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    filepath = os.path.join(script_dir, filename)    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Начальные условия для круговой орбиты
    h = 800e3  # Высота орбиты (м)
    r0 = R + h  # Радиус орбиты относительно начала отсчета (м)
    V0 = np.sqrt(mu / r0)  # Начальная скорость на круговой орбите

    # Начальные условия
    X0_s1 = np.array([r0, 0, 0, 0, V0, 0]) # 1-й спутник
    X0_s2 = np.array([r0, 0, 0, 0, 0, V0]) # 2-й спутник
    
    # Временной интервал для модели (10 витков ~ 90000 сек для низкой орбиты)
    t_span = (0, 10 * 2 * np.pi * np.sqrt(r0**3 / mu))

    # Собственное движение (без управления)
    sol_no_thrust_s1 = propagate_orbit(X0_s1, t_span)
    sol_no_thrust_s2 = propagate_orbit(X0_s2, t_span)
    
    print('##################################################')
    print(f'РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ – СОБСТВЕННОЕ ДВИЖЕНИЕ')
    print('##################################################')
    print(f'Точек интегрирования (без упр.): {len(sol_no_thrust_s1.t)}')

    print(f'Высота орбиты: {h/1000:.0f} (км)')
    print(f'Радиус орбиты: {r0/1000:.0f} (км)')
    print(f'Начальная скорость: {V0:.0f} (м/с)')
    print(f'Период обращения: {2*np.pi*np.sqrt(r0**3 / mu)/60:.1f} (мин)')
    print(f'Время моделирования: {t_span[1] / 60:.1f} (мин)')
    
    # Расстояния между спутниками    
    distance, close_approaches = satellite_distance(
        [sol_no_thrust_s1.y[0] / 1000, sol_no_thrust_s1.y[1] / 1000, sol_no_thrust_s1.y[2] / 1000],
        [sol_no_thrust_s2.y[0] / 1000, sol_no_thrust_s2.y[1] / 1000, sol_no_thrust_s2.y[2] / 1000],
        critical_distance / 1000,
        sol_no_thrust_s1.t,
        True,
        'approaches-no-control.txt'
        )    
    
    # 3D орбита
    fig1 = plot_orbit_3d(sol_no_thrust_s1, sol_no_thrust_s2)
    
    # Проекции орбиты
    fig2 = plot_orbit_projections(sol_no_thrust_s1)
    fig3 = plot_orbit_projections(sol_no_thrust_s2)
    
    t_span = sol_no_thrust_s1.t
    fig4 = plot_distance(t_span, distance, critical_distance / 1000)
    
    # Начальные условия
    X0_s1 = np.array([r0, 0, 0, 0, V0, 0]) # 1-й спутник
    X0_s2 = np.array([r0, 0, 0, 0, 0, V0]) # 2-й спутник
    
    # Временной интервал для модели (10 витков ~ 90000 сек для низкой орбиты)
    t_span = (0, 10 * 2 * np.pi * np.sqrt(r0**3 / mu))
    
    # Первый спутник с управлением
    sol_avoid_s1 = propagate_orbit(X0_s1, t_span,
                                   close_approaches=close_approaches,
                                   U=collision_avoidance_controller)
    # Второй спутник без управления
    sol_avoid_s2 = propagate_orbit(X0_s2, t_span, U=None)  # второй без управления    
    
    print('##################################################')
    print(f'РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ – УПРАВЛЯЕМОЕ ДВИЖЕНИЕ')
    print('##################################################')
    print(f'Точек интегрирования (без упр.): {len(sol_avoid_s1.t)}')

    print(f'Высота орбиты: {h/1000:.0f} (км)')
    print(f'Радиус орбиты: {r0/1000:.0f} (км)')
    print(f'Начальная скорость: {V0:.0f} (м/с)')
    print(f'Период обращения: {2*np.pi*np.sqrt(r0**3 / mu)/60:.1f} (мин)')
    print(f'Время моделирования: {t_span[1] / 60:.1f} (мин)')
    
    # Расстояния между спутниками    
    distance_avoid, close_approaches_avoid = satellite_distance(
        [sol_avoid_s1.y[0] / 1000, sol_avoid_s1.y[1] / 1000, sol_avoid_s1.y[2] / 1000],
        [sol_avoid_s2.y[0] / 1000, sol_avoid_s2.y[1] / 1000, sol_avoid_s2.y[2] / 1000],
        critical_distance / 1000,
        sol_avoid_s1.t,
        True,
        'approaches-control.txt'
        )
    
    fig5 = plot_orbit_3d(sol_avoid_s1, sol_avoid_s2)
    fig6 = plot_orbit_projections(sol_avoid_s1)
    
    t_span_avoid = sol_avoid_s1.t
    fig7 = plot_distance(t_span_avoid, distance_avoid, critical_distance / 1000)
    
    # Показываем все графики
    plt.show()
