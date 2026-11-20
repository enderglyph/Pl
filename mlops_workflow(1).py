import matplotlib.pyplot as plt

steps = ["Data Collection", "Data Preparation", "Model Training", "Model Evaluation",
         "Model Registry", "Deployment", "Monitoring", "Retraining"]

colors = ["lightblue", "lightgreen", "lightyellow", "lightpink",
          "plum", "paleturquoise", "lightsalmon", "pink"]

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
ax.text(5, 11.4, "MLOps Workflow", ha="center", va="center", fontsize=20, fontweight="bold")

for i, step in enumerate(steps):
    y = 10.5 - i
    ax.text(5, y, step, ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[i],
                      edgecolor="black", linewidth=1.5))
    if i < len(steps) - 1:
        ax.annotate("", xy=(5, y - 0.55), xytext=(5, y - 0.40),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))

plt.tight_layout()
plt.show()
