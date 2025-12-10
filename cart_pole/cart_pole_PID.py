import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current, dt):
        error = current - target

        # P term
        p_term = self.kp * error

        # I term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # D term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        self.prev_error = error

        return p_term + i_term + d_term

class CartPolePIDSim:
    def __init__(self):
        # 물리 파라미터
        self.M = 1.0; self.m = 0.1; self.l = 0.5; self.g = 9.8

        # 초기 상태: [x, x_dot, theta, theta_dot]
        # 5도(약 0.087 rad) 기울어진 상태에서 시작
        self.state = np.array([0.0, 0.0, 15 * np.pi / 180, 0.0])
        self.dt = 0.02

        # PID 제어기 설정 (튜닝 포인트!)
        # 목표: 각도(Theta)를 0으로 유지하라
        # kp=100: 조금만 기울어도 강하게 반응 (힘)
        # kd=20: 흔들림을 잡아주는 브레이크 역할 (매우 중요)
        self.pid_theta = PIDController(kp=100.0, ki=5.0, kd=20.0)

        # [참고] 위치 제어용 PID (카트가 너무 멀리 가지 않게 살짝 당겨줌)
        # 각도 제어가 훨씬 중요하므로 힘을 아주 약하게 설정 (kp=2.0)
        self.pid_x = PIDController(kp=20.0, ki=0.0, kd=20.0)

    def get_derivatives(self, state, force):
        # 물리 엔진 (이전과 동일)
        x, x_dot, theta, theta_dot = state

        sys_A = np.array([
            [self.M + self.m, self.m * self.l * np.cos(theta)],
            [self.m * self.l * np.cos(theta), self.m * self.l**2]
        ])
        sys_b = np.array([
            force + self.m * self.l * (theta_dot**2) * np.sin(theta),
            self.m * self.g * self.l * np.sin(theta)
        ])
        acc = np.linalg.solve(sys_A, sys_b)
        return np.array([x_dot, acc[0], theta_dot, acc[1]])

    def update(self):
        curr_x = self.state[0]
        curr_theta = self.state[2]

        # 1. 각도 제어 (Balance Control)
        # 폴이 오른쪽(+)으로 기울면 -> 카트를 오른쪽(+)으로 밀어야 함
        # 따라서 error = curr_theta (목표값 0)
        force_theta = self.pid_theta.compute(target=0.0, current=curr_theta, dt=self.dt)

        # 2. 위치 제어 (Position Control) - 보조 역할
        # 카트가 너무 오른쪽(+)으로 가면 -> 왼쪽(-)으로 힘을 줘야 함
        # 따라서 부호는 반대
        force_x = self.pid_x.compute(target=0.0, current=curr_x, dt=self.dt)

        # 최종 힘 = 중심 잡는 힘 + 위치 지키는 힘
        force = force_theta + force_x

        # 모터 힘 제한
        force = np.clip(force, -50, 50)

        # 물리 업데이트
        derivatives = self.get_derivatives(self.state, force)
        self.state += derivatives * self.dt

        return self.state, force

# --- 애니메이션 실행 ---
sim = CartPolePIDSim()

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.0, 2.0)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("PID Control: Balancing Pole")

cart_width = 0.6; cart_height = 0.3
cart = patches.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, fc='orange', ec='black')
pole, = ax.plot([], [], 'b-', linewidth=4)
ax.add_patch(cart)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    state, force = sim.update()
    x, _, theta, _ = state

    cart.set_xy((x - cart_width/2, -cart_height/2))
    pole_x = x + 2 * sim.l * np.sin(theta)
    pole_y = 2 * sim.l * np.cos(theta)
    pole.set_data([x, pole_x], [0, pole_y])
    time_text.set_text(f"Force: {force:.1f}N")
    return cart, pole, time_text

ani = animation.FuncAnimation(fig, animate, frames=300, interval=20, blit=True)
plt.show()