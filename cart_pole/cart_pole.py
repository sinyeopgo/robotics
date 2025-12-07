import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CartPoleSim:
    def __init__(self):
        # 물리 파라미터
        self.M = 1.0  # 카트 질량 (kg)
        self.m = 0.1  # 폴 질량 (kg)
        self.l = 0.5  # 폴 길이의 절반 (m)
        self.g = 9.8  # 중력 가속도

        # 상태 변수: [x, x_dot, theta, theta_dot]
        # 초기 상태: 폴이 약간 기울어진 상태 (0.1 rad)
        self.state = np.array([0.0, 0.0, 0.1, 0.0])
        self.dt = 0.02  # 시간 간격 (sec)

    def get_derivatives(self, state, force):
        x, x_dot, theta, theta_dot = state

        # 운동 방정식 (Matrix Form Ax = b)
        # 식 1: (M+m)x_dd + (ml cos)theta_dd = F + ml(theta_d^2)sin
        # 식 2: (ml cos)x_dd + (ml^2)theta_dd = mgl sin

        # A 행렬 (질량 관성 행렬)
        A = np.array([
            [self.M + self.m, self.m * self.l * np.cos(theta)],
            [self.m * self.l * np.cos(theta), self.m * self.l**2]
        ])

        # b 벡터 (코리올리 힘, 중력, 외력)
        b = np.array([
            force + self.m * self.l * (theta_dot**2) * np.sin(theta),
            self.m * self.g * self.l * np.sin(theta)
        ])

        # 가속도 구하기 (A * acc = b  => acc = A_inv * b)
        # x_ddot, theta_ddot
        acc = np.linalg.solve(A, b)

        return np.array([x_dot, acc[0], theta_dot, acc[1]])

    def step(self, force):
        # RK4 같은 정밀한 적분기 대신 이해를 돕기 위해 오일러 적분 사용
        # 다음 상태 = 현재 상태 + 미분값 * dt
        derivatives = self.get_derivatives(self.state, force)
        self.state += derivatives * self.dt
        return self.state

# --- 시뮬레이션 실행 및 애니메이션 ---
sim = CartPoleSim()
history = []

# 5초간 시뮬레이션 (힘은 0)
for _ in range(250):
    state = sim.step(force=0.0)
    history.append(state.copy())

history = np.array(history)

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')
cart_width = 0.4
cart_height = 0.2

cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, fc='black')
pole_line, = ax.plot([], [], 'r-', linewidth=3)
ax.add_patch(cart_rect)

def animate(i):
    x = history[i, 0]
    theta = history[i, 2]

    # 카트 위치 업데이트
    cart_rect.set_xy((x - cart_width/2, 0))

    # 폴 끝 위치 계산
    pole_x = x + 2 * sim.l * np.sin(theta)
    pole_y = cart_height + 2 * sim.l * np.cos(theta)

    pole_line.set_data([x, pole_x], [cart_height, pole_y])
    return cart_rect, pole_line

ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=20, blit=True)
plt.title("Phase 1: Free Fall Simulation (No Control)")
plt.grid()
plt.show()