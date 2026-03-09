import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


def satellite_model(t, X, U=None):
    """
    Функция правых частей модели спутника
    
    Параметры:
    :param t: float - время
    :param X: array - вектор состояния [x, y, z, Vx, Vy, Vz]
    :param U: array - управление [ux, uy, uz] (нормированное)
    
    Возвращаемое значение:
    :return dX/dt: array - производная вектора состояния
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

def propagate_orbit(X0, t_span, U=None, method='DOP853', rtol=1e-10, atol=1e-13, max_step=60):
    """
    Функция расчета орбиты спутника

    Параметры:
    :param X0: array - вектор состояния [x, y, z, Vx, Vy, Vz]
    :param t_span: tuple - интервал времени моделирования (t_start, t_end)
    :param U: array - управление [ux, uy, uz] (нормированное)
    :param method: str - метод интегрирования (по умолчанию Рунге-Кутта 8-го порядка)
    :param rtol: float - относительная точность
    :param atol: float - абсолютная точность
    :param max_step: float - максимальный шаг интегрирования (сек)

    Возвращаемое значение:
    :return solution: OdeSolution - решение на интервале
    """

    def satellite_dynamics(t, X):
        if U is None:
            U_local = [0, 0, 0]
        else:
            U_local = U(t)
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


def plot_orbit_3d(sol, show_earth=True):
    """
    Построение 3D графика орбиты
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Координаты центра масс спутника
    x = sol.y[0] / 1000  # Переводим в км.
    y = sol.y[1] / 1000
    z = sol.y[2] / 1000
    
    # Строим орбиту
    ax.plot(x, y, z, 'b-', linewidth=2, label='Орбита спутника')
    
    # Отмечаем начальную точку
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, marker='o')
    
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
    max_range = max(np.max(x) - np.min(x), 
                   np.max(y) - np.min(y), 
                   np.max(z) - np.min(z)) / 2
    mid_x = (np.max(x) + np.min(x)) / 2
    mid_y = (np.max(y) + np.min(y)) / 2
    mid_z = (np.max(z) + np.min(z)) / 2
    
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
    # ax.scatter(x[0], y[0], color='green', s=50, label='Начало')
    # ax.scatter(x[-1], y[-1], color='red', s=50, label='Конец')
    
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
    # ax.scatter(x[0], z[0], color='green', s=50)
    # ax.scatter(x[-1], z[-1], color='red', s=50)
    
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
    # ax.scatter(y[0], z[0], color='green', s=50)
    # ax.scatter(y[-1], z[-1], color='red', s=50)
    
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


if __name__ == "__main__":
    # Начальные условия для круговой орбиты
    h = 800e3  # Высота орбиты (м)
    r0 = R + h  # Радиус орбиты
    V0 = np.sqrt(mu / r0)  # Скорость для круговой орбиты

    # Начальные условия для модели
    X0 = np.array([r0, 0, 0, 0, V0, 0])
    # Временной интервал для модели (10 витков ~ 90000 сек для низкой орбиты)
    t_span = (0, 10 * 2 * np.pi * np.sqrt(r0**3 / mu))

    

    print(f'Высота орбиты: {h/1000:.0f} (км)')
    print(f'Период обращения: {2*np.pi*np.sqrt(r0**3 / mu)/60:.1f} (мин)')
    print(f'Время моделирования: {t_span[1]/60:.1f} (мин)')

    # Собственное движение (без управления)
    sol_no_thrust = propagate_orbit(X0, t_span)
    print(f'Результаты:')
    print(f'Точек интегрирования (без упр.): {len(sol_no_thrust.t)}')

    # 3D орбита
    fig1 = plot_orbit_3d(sol_no_thrust)
    
    # Проекции орбиты
    fig2 = plot_orbit_projections(sol_no_thrust)    
    
    # Показываем все графики
    plt.show()
