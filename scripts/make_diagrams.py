"""Renders the two block diagrams the report and slides need, as PNG + vector PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = "results/figures"
INK = "#1b2733"
ACCENT = "#1f4e79"
WARM = "#9c3d1e"


def box(ax, x, y, w, h, title, lines, edge=ACCENT, face="#f4f7fa"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                linewidth=1.3, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h - 0.040, title, ha="center", va="top",
            fontsize=10.0, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + 0.028, y + h - 0.098 - i * 0.040, ln, ha="left", va="top",
                fontsize=7.7, color="#44525f")


def arrow(ax, x, y0, y1, label, side="left", color=ACCENT):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.2, color=color))
    ha = "right" if side == "left" else "left"
    dx = -0.014 if side == "left" else 0.014
    ax.text(x + dx, (y0 + y1) / 2, label, ha=ha, va="center", fontsize=7.3,
            color=color)


def control_structure():
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.15, 0.745, 0.70, 0.215, "TIER 1   Fleet Safety Management",
        ["Route and dispatch policy, speed policy, ODD limits",
         "Accountable for the operational design domain",
         "Receives interlock audit logs and exceedance trends"])
    box(ax, 0.15, 0.395, 0.70, 0.255,
        "TIER 2   Vehicle Visibility Supervisor",
        ["Software controller enforcing SC-1 to SC-5",
         "Process model: sight distance, particulate load,",
         "road class, commanded speed, override state",
         "AI severity prediction is an input, never an authority"],
        edge=WARM, face="#fdf5f1")
    box(ax, 0.15, 0.055, 0.70, 0.215, "TIER 3   Actuators and driver",
        ["Engine torque limit, service brake, hazard lamps",
         "Driver override switch has priority (SC-3)",
         "Reports wheel speed and achieved deceleration"])
    arrow(ax, 0.32, 0.742, 0.655, "speed policy, ODD limits", "left")
    arrow(ax, 0.68, 0.655, 0.742, "audit log, exceedances", "right", "#5b6b7a")
    arrow(ax, 0.32, 0.392, 0.275, "target speed, hazard command", "left", WARM)
    arrow(ax, 0.68, 0.275, 0.392, "visibility and PM readings", "right", "#5b6b7a")
    ax.text(0.5, 0.998, "STAMP hierarchical control structure", ha="center", va="top",
            fontsize=11.6, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/control_structure.png", dpi=300)
    fig.savefig(f"{FIG}/control_structure.pdf")
    plt.close(fig)


def data_flow():
    fig, ax = plt.subplots(figsize=(9.8, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, STEP, X0 = 0.158, 0.199, 0.012
    stages = [
        ("STATS19\n2021-2025", ["collision 513,801", "vehicle 937,265",
                                 "casualty 652,821"]),
        ("Link on\ncollision_index", ["1:N vehicle", "1:N casualty",
                                      "goods-vehicle flag"]),
        ("Quality\nrepair", ["sentinel -1 to NaN", "boundary rules",
                             "median imputation"]),
        ("AURN\nparticulates", ["87 PM10 sites", "157,402 site-days",
                                "nearest-site join"]),
        ("Analysis\npanels", ["74,804 goods", "1,826 days",
                              "collision + daily"]),
    ]
    for i, (title, lines) in enumerate(stages):
        x = X0 + i * STEP
        ax.add_patch(FancyBboxPatch((x, 0.20), W, 0.62,
                                    boxstyle="round,pad=0.008", linewidth=1.25,
                                    edgecolor=ACCENT, facecolor="#f4f7fa"))
        ax.text(x + W / 2, 0.775, title, ha="center", va="top", fontsize=9.0,
                fontweight="bold", color=INK, linespacing=1.35)
        for j, ln in enumerate(lines):
            ax.text(x + W / 2, 0.475 - j * 0.082, ln, ha="center", va="top",
                    fontsize=7.2, color="#44525f")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.006, 0.51),
                                         (x + STEP - 0.006, 0.51),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.2, color="#5b6b7a"))
    ax.text(0.5, 0.985, "Ingestion and information-quality pipeline", ha="center",
            va="top", fontsize=11.0, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(f"{FIG}/data_flow.png", dpi=300)
    fig.savefig(f"{FIG}/data_flow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    control_structure()
    data_flow()
    print(f"wrote {FIG}/control_structure.png/.pdf and {FIG}/data_flow.png/.pdf")
