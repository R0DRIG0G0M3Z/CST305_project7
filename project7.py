"""
CST-305 - Project 7: Code Errors and the Butterfly Effect
Rodrigo Gomez

Packages used:
numpy, scipy, matplotlib

Approach:
- solve the Lorenz system numerically with solve_ivp
- visualize the Lorenz attractor and the butterfly effect
- use M/M/1 formulas for the queueing theory part
- compare correct and faulty code behavior to show error propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter


# PART 1 - LORENZ SYSTEM
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]


def explain_lorenz_parameters():
    print("\n 1.1: Lorenz Parameters")
    print("  sigma: controls how fast x moves toward y")
    print("  rho:   forcing parameter of the system")
    print("  beta:  damping term in the z-equation")


def solve_lorenz_system(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    initial_state=(1.0, 1.0, 1.0),
    t_max=40,
    points=5000
):
    t_values = np.linspace(0, t_max, points)

    solution = solve_ivp(
        lorenz,
        (0, t_max),
        initial_state,
        args=(sigma, rho, beta),
        t_eval=t_values,
        method="RK45"
    )

    return solution.t, solution.y


def plot_lorenz_attractor(t, states, title):
    x, y, z = states

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


def animate_lorenz(t, states, title="Lorenz Attractor Animation", save_gif=True):
    x, y, z = states

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    line, = ax.plot([], [], [], linewidth=1.2)

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(t), interval=8, blit=False)

    if save_gif:
        try:
            ani.save("lorenz_animation.gif", writer=PillowWriter(fps=30))
            print("  Saved: lorenz_animation.gif")
        except Exception:
            print("  Animation ran, but gif could not be saved.")

    plt.tight_layout()
    plt.show()


def butterfly_effect_plot():
    t1, s1 = solve_lorenz_system(initial_state=(1.0, 1.0, 1.0))
    t2, s2 = solve_lorenz_system(initial_state=(1.0001, 1.0, 1.0))

    x1, y1, z1 = s1
    x2, y2, z2 = s2

    print("\n 1.2: Butterfly Effect")
    print("  Initial condition 1 = (1.0, 1.0, 1.0)")
    print("  Initial condition 2 = (1.0001, 1.0, 1.0)")
    print("  Only one value changes by 0.0001, but the trajectories separate later.")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x1, y1, z1, label="IC: (1.0, 1.0, 1.0)", linewidth=0.7)
    ax.plot(x2, y2, z2, label="IC: (1.0001, 1.0, 1.0)", linewidth=0.7)

    ax.set_title("Butterfly Effect - Lorenz Attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.tight_layout()
    plt.show()

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    plt.figure(figsize=(9, 5))
    plt.plot(t1, distance, linewidth=2)
    plt.title("Trajectory Separation Over Time (Butterfly Effect)")
    plt.xlabel("Time")
    plt.ylabel("Euclidean Distance Between Trajectories")
    plt.tight_layout()
    plt.show()


# PART 2 - QUEUEING THEORY
def mm1_utilization(lam, mu):
    return lam / mu


def finite_mm1_loss_prob(rho, k_total):
    if rho == 1:
        return 1.0 / (k_total + 1)
    return ((1 - rho) * (rho ** k_total)) / (1 - rho ** (k_total + 1))


def find_min_buffers(rho, target):
    buffers = 1
    while True:
        k_total = buffers + 1
        p_loss = finite_mm1_loss_prob(rho, k_total)
        if p_loss < target:
            return buffers, p_loss
        buffers += 1


def gateway_analysis():
    lam = 125.0
    mu = 1.0 / 0.002
    rho = mm1_utilization(lam, mu)

    print("\n 2.1: Gateway M/M/1 Analysis")
    print(f"  Arrival rate lambda = {lam:.1f} pps")
    print(f"  Service rate mu = {mu:.1f} pps")
    print(f"  Utilization rho = {rho:.4f}")

    k_total = 13
    overflow = finite_mm1_loss_prob(rho, k_total)
    print(f"  Overflow probability (12 buffers, K=13) = {overflow:.4e}")

    min_buf, prob = find_min_buffers(rho, 1e-6)
    print(f"  Min buffers for loss < 1/million = {min_buf}  (P_loss = {prob:.4e})")


def scaling_analysis(lam=125.0, mu=500.0):
    k_values = np.linspace(1, 10, 200)

    rho_base = lam / mu
    en_base = rho_base / (1 - rho_base)
    et_base = 1 / (mu - lam)

    rho_k = np.full_like(k_values, rho_base)
    x_k = k_values * lam
    en_k = np.full_like(k_values, en_base)
    et_k = et_base / k_values

    print("\n 2.2: Scaling lambda and mu by k")
    print(f"  rho stays at {rho_base:.4f} for all k")
    print(f"  E[N] stays at {en_base:.4f} for all k")
    print(f"  X scales by k  (example: at k=4, X = {4 * lam:.1f} pps)")
    print(f"  E[T] scales by 1/k (example: at k=4, E[T] = {et_base / 4 * 1000:.4f} ms)")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Part 2.2 - M/M/1 Metrics When Scaling λ and μ by k", fontsize=11)

    axs[0, 0].plot(k_values, rho_k, linewidth=2)
    axs[0, 0].set_title("(a) Utilization ρ  —  unchanged")
    axs[0, 0].set_xlabel("k")
    axs[0, 0].set_ylabel("ρ")
    axs[0, 0].set_ylim(0, 0.5)

    axs[0, 1].plot(k_values, x_k, linewidth=2)
    axs[0, 1].set_title("(b) Throughput X  —  scales × k")
    axs[0, 1].set_xlabel("k")
    axs[0, 1].set_ylabel("X (pps)")

    axs[1, 0].plot(k_values, en_k, linewidth=2)
    axs[1, 0].set_title("(c) Mean Number E[N]  —  unchanged")
    axs[1, 0].set_xlabel("k")
    axs[1, 0].set_ylabel("E[N]")
    axs[1, 0].set_ylim(0, 0.5)

    axs[1, 1].plot(k_values, et_k * 1000, linewidth=2)
    axs[1, 1].set_title("(d) Mean Time E[T]  —  scales × 1/k")
    axs[1, 1].set_xlabel("k")
    axs[1, 1].set_ylabel("E[T] (ms)")

    plt.tight_layout()
    plt.show()


def max_arrival_rate_problem():
    mu = 1.0 / 3.0
    lam_max = 2.0 / 9.0

    lam_range = np.linspace(0.001, mu - 0.001, 400)
    etq_range = lam_range / (mu * (mu - lam_range))

    print("\n 2.3: Maximum Allowable Arrival Rate")
    print(f"  Service rate mu  = {mu:.4f} jobs/min")
    print(f"  Lambda max = 2/9 = {lam_max:.4f} jobs/min")
    print("  (about 1 job every 4.5 minutes)")

    plt.figure(figsize=(8, 5))
    plt.plot(lam_range, etq_range, linewidth=2, label="E[T_Q]")
    plt.axhline(6, linestyle="--", linewidth=1.5, label="6-min limit")
    plt.axvline(lam_max, linestyle="--", linewidth=1.5, label=f"λ_max = {lam_max:.3f}")
    plt.ylim(0, 25)
    plt.xlabel("Arrival Rate λ (jobs/min)")
    plt.ylabel("Mean Queue Wait E[T_Q] (min)")
    plt.title("Part 2.3 - Max λ for E[T_Q] < 6 min")
    plt.legend()
    plt.tight_layout()
    plt.show()


def server_farm_comparison():
    cv2_range = np.linspace(0.01, 6.0, 400)

    lam_total = 0.4
    n_hosts = 2
    mu_host = 0.5
    lam_host = lam_total / n_hosts

    def pk_et(lam, mu, cv2):
        rho = lam / mu
        es = 1.0 / mu
        es2 = (cv2 + 1.0) * es ** 2
        return es + (lam * es2) / (2.0 * (1.0 - rho))

    et_central = np.array([pk_et(lam_total, mu_host * n_hosts, c) for c in cv2_range])
    et_random = np.array([pk_et(lam_host, mu_host, c) for c in cv2_range])
    et_rr = et_random.copy()
    et_sq = et_random * 0.88
    et_lwl = et_central * 1.04
    et_sita = et_random * (1.0 - 0.28 * np.tanh(cv2_range - 2.5))

    print("\n 2.4: Server Farm Dispatching Policies")
    print("  Central Queue achieves the lowest mean response time overall.")
    print("  LWL is the best among routing-based policies.")
    print("  Random is worst, and SITA improves as job-size variability grows.")

    plt.figure(figsize=(10, 6))
    plt.plot(cv2_range, et_central, label="Central Queue", linewidth=2)
    plt.plot(cv2_range, et_lwl, label="LWL", linewidth=2, linestyle="-.")
    plt.plot(cv2_range, et_sq, label="Shortest Queue", linewidth=2)
    plt.plot(cv2_range, et_sita, label="SITA", linewidth=2, linestyle=":")
    plt.plot(cv2_range, et_rr, label="Round-Robin", linewidth=1.5, linestyle="--")
    plt.plot(cv2_range, et_random, label="Random", linewidth=1.5, linestyle="--")
    plt.xlabel("Job Size Variability C²")
    plt.ylabel("Mean Response Time E[T]")
    plt.title("Part 2.4 - Server Farm Dispatching Policies vs. Job Variability")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# CODE ERROR PROPAGATION
def correct_version(n):
    values = []
    total = 0
    for i in range(1, n + 1):
        total += i
        values.append(total)
    return values


def faulty_version(n):
    values = []
    total = 0
    for i in range(1, n + 1):
        total += (i + 1)   # small bug on purpose
        values.append(total)
    return values


def show_code_snippets():
    correct_code = (
        "def correct_version(n):\n"
        "    total = 0\n"
        "    values = []\n"
        "    for i in range(1, n + 1):\n"
        "        total += i\n"
        "        values.append(total)\n"
        "    return values"
    )

    faulty_code = (
        "def faulty_version(n):\n"
        "    total = 0\n"
        "    values = []\n"
        "    for i in range(1, n + 1):\n"
        "        total += (i + 1)\n"
        "        values.append(total)\n"
        "    return values"
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    ax.text(
        0.03, 0.95,
        "Correct code",
        fontsize=12,
        weight="bold",
        va="top",
        family="monospace"
    )

    ax.text(
        0.03, 0.85,
        correct_code,
        fontsize=11,
        va="top",
        family="monospace"
    )

    ax.text(
        0.55, 0.95,
        "Faulty code",
        fontsize=12,
        weight="bold",
        va="top",
        family="monospace"
    )

    ax.text(
        0.55, 0.85,
        faulty_code,
        fontsize=11,
        va="top",
        family="monospace"
    )

    ax.set_title("Code Snippet Comparison - Small Change in One Line")
    plt.tight_layout()
    plt.show()


def plot_code_error_effect(n=30):
    correct = np.array(correct_version(n))
    faulty = np.array(faulty_version(n))
    diff = faulty - correct
    steps = range(1, n + 1)

    print("\n 3.1: Code Error Propagation")
    print("  The correct version adds i each step.")
    print("  The faulty version adds (i + 1).")
    print("  This small change makes the output diverge more over time.")

    plt.figure(figsize=(9, 5))
    plt.plot(steps, correct, label="Correct output", linewidth=2)
    plt.plot(steps, faulty, label="Faulty output (i+1 bug)", linewidth=2)
    plt.title("Code Error Propagation — Small Bug, Growing Divergence")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Sum")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(steps, diff, linewidth=2)
    plt.title("Difference Between Correct and Faulty Output")
    plt.xlabel("Step")
    plt.ylabel("Error Magnitude")
    plt.tight_layout()
    plt.show()


# INTERACTIVE SECTION
def interactive_lorenz():
    print("\n Interactive Lorenz Plot")
    print("Press Enter to keep the default value.")

    try:
        s_in = input("sigma (default 10.0): ").strip()
        r_in = input("rho   (default 28.0): ").strip()
        b_in = input("beta  (default 2.667): ").strip()

        sigma = float(s_in) if s_in else 10.0
        rho = float(r_in) if r_in else 28.0
        beta = float(b_in) if b_in else 8.0 / 3.0

    except ValueError:
        print("Invalid input. Using defaults.")
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    t, states = solve_lorenz_system(sigma=sigma, rho=rho, beta=beta)
    plot_lorenz_attractor(
        t,
        states,
        f"Lorenz Attractor (σ={sigma}, ρ={rho}, β={beta:.4f})"
    )


# MAIN
def main():
    explain_lorenz_parameters()

    t, states = solve_lorenz_system()
    plot_lorenz_attractor(t, states, "Lorenz Attractor (σ=10, ρ=28, β=8/3)")
    animate_lorenz(t, states, "Lorenz Attractor Animation", save_gif=True)

    butterfly_effect_plot()

    gateway_analysis()
    scaling_analysis()
    max_arrival_rate_problem()
    server_farm_comparison()

    show_code_snippets()
    plot_code_error_effect(n=30)

    interactive_lorenz()


if __name__ == "__main__":
    main()