"""
Outlier Detection and Handling Using IQR and Z-score
Run:
    python -m pip install numpy matplotlib pillow
    python outlier_detection_animation.py

Creates:
    outlier_detection_animation.gif
    outlier_animation_preview.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

DATA = np.array([45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                 56, 57, 58, 59, 60, 61, 62, 63, 65, 98], dtype=float)

# Median-of-halves quartiles (explicitly chosen for reproducibility)
lower_half, upper_half = DATA[:len(DATA)//2], DATA[len(DATA)//2:]
q1 = (lower_half[4] + lower_half[5]) / 2
q3 = (upper_half[4] + upper_half[5]) / 2
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

mean = DATA.mean()
sd = DATA.std(ddof=0)  # population standard deviation
z_scores = (DATA - mean) / sd
z_outliers = np.abs(z_scores) > 3
iqr_outliers = (DATA < lower_fence) | (DATA > upper_fence)

# Demonstration handling: cap flagged values at the IQR upper fence.
capped = np.clip(DATA, lower_fence, upper_fence)

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(10, 5.8))
fig.subplots_adjust(top=0.82, bottom=0.15, left=0.09, right=0.97)
frames_per_scene = 35
scene_titles = [
    "1. Meet the dataset",
    "2. IQR: find the middle 50%",
    "3. IQR: flag values outside the fences",
    "4. Z-score: measure distance from the mean",
    "5. Z-score: flag |z| > 3",
    "6. Handling: compare original and capped data",
    "7. Key takeaway"
]

def base(title, subtitle):
    ax.clear()
    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Student mark")
    ax.set_xlim(35, 105)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.22)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#444444")

def scatter_values(y=0.5, values=DATA, colors=None, size=75):
    if colors is None:
        colors = ["#4776A8"] * len(values)
    ax.scatter(values, np.full(len(values), y), c=colors, s=size,
               edgecolors="white", linewidths=0.7, zorder=3)

def animate(frame):
    scene = min(frame // frames_per_scene, len(scene_titles)-1)
    progress = (frame % frames_per_scene) / (frames_per_scene - 1)
    title = scene_titles[scene]

    if scene == 0:
        base(title, "Most marks cluster together; 98 is far from the rest.")
        colors = ["#4776A8"] * len(DATA)
        if progress > 0.55:
            colors[-1] = "#D94F4F"
        scatter_values(values=DATA, colors=colors, size=95)
        ax.set_ylim(0, 1)
        ax.text(98, 0.68, "98", ha="center", color="#D94F4F", weight="bold")
        ax.text(0.02, 0.82, "Dataset: " + ", ".join(map(str, DATA.astype(int))),
                transform=ax.transAxes, fontsize=8, wrap=True)
        ax.text(0.02, 0.12, "Question: is 98 unusual, and what should we do with it?",
                transform=ax.transAxes, fontsize=10)

    elif scene == 1:
        base(title, "IQR = Q3 − Q1; the box represents the middle half of the data.")
        ax.set_xlim(35, 105)
        ax.set_ylim(0, 1)
        ax.plot([DATA.min(), DATA.max()], [0.45, 0.45], color="#AAAAAA", lw=2)
        ax.add_patch(Rectangle((q1, 0.28), q3-q1, 0.34,
                               facecolor="#CFE4F5", edgecolor="#245B85", lw=2))
        ax.plot([np.median(DATA), np.median(DATA)], [0.28, 0.62],
                color="#245B85", lw=2)
        for x, label in [(q1, f"Q1 = {q1:.1f}"),
                         (np.median(DATA), f"Median = {np.median(DATA):.1f}"),
                         (q3, f"Q3 = {q3:.1f}")]:
            ax.plot([x, x], [0.22, 0.68], color="#245B85", ls="--", alpha=0.8)
            ax.text(x, 0.76, label, ha="center", fontsize=9)
        ax.text(0.5, 0.08, f"IQR = {q3:.1f} − {q1:.1f} = {iqr:.1f}",
                transform=ax.transAxes, ha="center", fontsize=12, weight="bold")

    elif scene == 2:
        base(title, "Flag a value if it is below the lower fence or above the upper fence.")
        ax.set_ylim(0, 1)
        ax.axvspan(lower_fence, upper_fence, color="#CDE8D1", alpha=0.7,
                   label="Inside IQR fences")
        ax.axvline(lower_fence, color="#37834A", ls="--", lw=2)
        ax.axvline(upper_fence, color="#37834A", ls="--", lw=2)
        colors = ["#D94F4F" if x < lower_fence or x > upper_fence else "#4776A8" for x in DATA]
        scatter_values(values=DATA, colors=colors, size=90)
        ax.text(lower_fence, 0.72, f"Lower = {lower_fence:.1f}", ha="center", fontsize=9)
        ax.text(upper_fence, 0.72, f"Upper = {upper_fence:.1f}", ha="center", fontsize=9)
        ax.annotate("98 flagged", xy=(98, 0.5), xytext=(89, 0.82),
                    arrowprops=dict(arrowstyle="->", color="#D94F4F"),
                    color="#D94F4F", weight="bold")
        ax.text(0.03, 0.10, "IQR result: 98 is above the upper fence.",
                transform=ax.transAxes, fontsize=11, weight="bold")

    elif scene == 3:
        base(title, "Z-score tells how many standard deviations a value is from the mean.")
        ax.set_xlim(35, 105)
        ax.set_ylim(0, 1)
        ax.axvline(mean, color="#245B85", lw=2)
        ax.axvline(mean + sd, color="#777777", ls="--")
        ax.axvline(mean - sd, color="#777777", ls="--")
        ax.axvline(mean + 3*sd, color="#D94F4F", ls=":")
        scatter_values(values=DATA, colors=["#D94F4F" if x == 98 else "#4776A8" for x in DATA])
        ax.text(mean, 0.77, f"Mean = {mean:.2f}", ha="center", color="#245B85", weight="bold")
        ax.text(mean+sd, 0.60, "+1 SD", ha="center", fontsize=9)
        ax.text(mean-sd, 0.60, "−1 SD", ha="center", fontsize=9)
        ax.text(mean+3*sd, 0.77, "+3 SD", ha="center", color="#D94F4F", fontsize=9)
        ax.annotate(f"98 is {z_scores[-1]:.2f} SD above mean",
                    xy=(98, 0.5), xytext=(76, 0.18),
                    arrowprops=dict(arrowstyle="->", color="#D94F4F"),
                    color="#D94F4F", fontsize=10, weight="bold")

    elif scene == 4:
        base(title, "A common rule flags observations whose absolute Z-score exceeds 3.")
        ax.set_ylim(0, 1)
        colors = ["#D94F4F" if flag else "#4776A8" for flag in z_outliers]
        scatter_values(values=DATA, colors=colors, size=90)
        ax.axvline(mean, color="#245B85", lw=2, label="Mean")
        ax.axvline(mean+3*sd, color="#D94F4F", ls="--", lw=2, label="Mean + 3 SD")
        ax.text(0.03, 0.82, f"Mean = {mean:.2f}   Population SD = {sd:.2f}",
                transform=ax.transAxes, fontsize=10)
        ax.text(0.03, 0.13, f"z(98) = (98 − {mean:.2f}) / {sd:.2f} = {z_scores[-1]:.2f}",
                transform=ax.transAxes, fontsize=11, weight="bold")
        ax.text(0.65, 0.82, "98 flagged because |z| > 3",
                transform=ax.transAxes, color="#D94F4F", fontsize=10, weight="bold")

    elif scene == 5:
        base(title, "Example strategy: cap the flagged value at the upper IQR fence.")
        ax.set_ylim(0, 1)
        ax.scatter(DATA, np.full(len(DATA), 0.68), c="#4776A8", s=70,
                   edgecolors="white", label="Original")
        ax.scatter(capped, np.full(len(capped), 0.30), c="#37834A", s=70,
                   edgecolors="white", label="Capped at IQR fences")
        ax.axvline(upper_fence, color="#37834A", ls="--", lw=2)
        ax.text(98, 0.78, "Original 98", ha="center", color="#D94F4F", weight="bold")
        ax.text(upper_fence, 0.12, f"Capped to {upper_fence:.1f}",
                ha="center", color="#37834A", weight="bold")
        ax.text(0.03, 0.04, "Other options: investigate and retain, correct an error, or remove with justification.",
                transform=ax.transAxes, fontsize=8.5)

    else:
        base(title, "Detection is a flag for investigation—not an automatic instruction to delete.")
        ax.set_ylim(0, 1)
        ax.axis("off")
        items = [
            ("IQR", f"Fences: {lower_fence:.1f} to {upper_fence:.1f}"),
            ("Z-score", f"z(98) = {z_scores[-1]:.2f}; threshold = 3"),
            ("Result", "Both methods flag 98 in this example."),
            ("Handling", "Check context before correcting, capping, removing, or retaining.")
        ]
        for i, (head, body) in enumerate(items):
            y = 0.78 - i*0.19
            ax.text(0.08, y, head, transform=ax.transAxes, fontsize=12,
                    weight="bold", color="#245B85",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#EAF2F8", ec="#245B85"))
            ax.text(0.30, y, body, transform=ax.transAxes, fontsize=11, va="center")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

total_frames = frames_per_scene * len(scene_titles)
anim = FuncAnimation(fig, animate, frames=total_frames, interval=90, repeat=True)
anim.save("outlier_detection_animation.gif", writer=PillowWriter(fps=12))
animate(2 * frames_per_scene + 10)
fig.savefig("outlier_animation_preview.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("Saved outlier_detection_animation.gif and outlier_animation_preview.png")
