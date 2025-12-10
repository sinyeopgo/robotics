import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg as linalg  # 리카티 방정식 풀이를 위해 추가

class CartPoleSim:
    def __init__(self):
        # 물리 파라미터
        self.M = 1.0  # 카트 질량 (kg)
        self.m = 0.1  # 폴 질량 (kg)
        self.l = 0.5  # 폴 길이 (회전축에서 무게중심까지의 거리)
        self.g = 9.8  # 중력 가속도

        # 상태 변수: [x, x_dot, theta, theta_dot]
        # 초기 상태: 폴이 약간 기울어진 상태 (0.1 rad)
        self.state = np.array([0.0, 0.0, 0.1, 0.0])
        self.dt = 0.02  # 시간 간격 (sec)

    def get_derivatives(self, state, force):
        x, x_dot, theta, theta_dot = state

        # 비선형 운동 방정식 (시뮬레이션 용 - 실제 물리 거동)
        # 식 1: (M+m)x_dd + (ml cos)theta_dd = F + ml(theta_d^2)sin
        # 식 2: (ml cos)x_dd + (ml^2)theta_dd = mgl sin

        # A_mat (질량 관성 행렬)
        A_mat = np.array([
            [self.M + self.m, self.m * self.l * np.cos(theta)],
            [self.m * self.l * np.cos(theta), self.m * self.l**2]
        ])

        # b 벡터 (코리올리 힘, 중력, 외력)
        b = np.array([
            force + self.m * self.l * (theta_dot**2) * np.sin(theta),
            self.m * self.g * self.l * np.sin(theta)
        ])

        # 가속도 구하기
        acc = np.linalg.solve(A_mat, b)
        return np.array([x_dot, acc[0], theta_dot, acc[1]])

    def step(self, force):
        derivatives = self.get_derivatives(self.state, force)
        self.state += derivatives * self.dt
        return self.state

    # --- LQR 설계를 위한 선형 모델 및 게인 계산 함수 추가 ---
    def compute_lqr_gain(self):
        # 평형점(theta=0) 근처에서의 선형화 모델 유도 (Small Angle Approximation)
        # 선형화된 운동 방정식:
        # 1. (M+m)x_dd + ml*theta_dd = u
        # 2. ml*x_dd + ml^2*theta_dd = mgl*theta

        # 위 식을 정리하여 x_dot = Ax + Bu 형태로 변환하면 아래와 같습니다.
        # (직접 유도하거나 표준 도립진자 수식을 참고)

        # 시스템 행렬 A (4x4)
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, (self.M + self.m) * self.g / (self.M * self.l), 0]
        ])

        # 입력 행렬 B (4x1)
        B = np.array([
            [0],
            [1 / self.M],
            [0],
            [-1 / (self.M * self.l)]
        ])

        # 가중치 행렬 Q (상태 중요도)
        # [위치, 속도, 각도, 각속도]
        Q = np.diag([1.0, 1.0, 100.0, 1.0])

        # 가중치 행렬 R (입력 비용)
        R = np.array([[0.1]])

        # 리카티 방정식 풀이 (Continuous Algebraic Riccati Equation)
        P = linalg.solve_continuous_are(A, B, Q, R)

        # 최적 게인 K 계산
        K = np.linalg.inv(R) @ B.T @ P

        print(f"Calculated LQR Gain K: {K}")
        return K

# --- 시뮬레이션 실행 ---
sim = CartPoleSim()

# 1. LQR 게인 미리 계산
K = sim.compute_lqr_gain()

history = []
force_history = []
time_steps = 300  # 6초

for i in range(time_steps):
    # 2. 현재 상태 측정
    current_state = sim.state

    # 목표 상태는 모두 0 (수직, 정지, 원점)
    # 오차 = current_state - target_state (target은 0이므로 생략)

    # 3. LQR 제어 입력 계산: u = -Kx
    # K는 (1,4), state는 (4,), 결과는 스칼라
    force = -np.dot(K, current_state).item()

    # (선택) 실제 모터 한계(Saturation) 적용
    max_force = 20.0
    force = np.clip(force, -max_force, max_force)

    # 4. 시뮬레이션 스텝 진행
    state = sim.step(force)

    history.append(state.copy())
    force_history.append(force)

history = np.array(history)

# --- 애니메이션 및 결과 시각화 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})

# 1. 애니메이션 설정
ax1.set_xlim(-3, 3)
ax1.set_ylim(-1, 2)
ax1.set_aspect('equal')
ax1.set_title("LQR Controlled Inverted Pendulum")
ax1.grid()

cart_width = 0.6
cart_height = 0.3
cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, fc='black')
pole_line, = ax1.plot([], [], 'r-', linewidth=4)
ax1.add_patch(cart_rect)

# 시간 표시 텍스트
time_template = 'Time = %.1fs'
time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

# 2. 힘(Force) 그래프 설정
ax2.set_xlim(0, time_steps * sim.dt)
ax2.set_ylim(-25, 25)
ax2.set_ylabel('Force (N)')
ax2.set_xlabel('Time (s)')
ax2.grid()
force_line, = ax2.plot([], [], 'b-')

def animate(i):
    # 물리 상태 업데이트
    x = history[i, 0]
    theta = history[i, 2]

    cart_rect.set_xy((x - cart_width/2, 0))
    pole_x = x + 2 * sim.l * np.sin(theta)
    pole_y = cart_height + 2 * sim.l * np.cos(theta)
    pole_line.set_data([x, pole_x], [cart_height, pole_y])

    time_text.set_text(time_template % (i * sim.dt))

    # 힘 그래프 업데이트
    force_line.set_data(np.arange(i) * sim.dt, force_history[:i])

    return cart_rect, pole_line, time_text, force_line

ani = animation.FuncAnimation(fig, animate, frames=len(history), interval=sim.dt*1000, blit=True)
plt.tight_layout()
plt.show()