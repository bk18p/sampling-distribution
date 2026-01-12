import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

APP_VERSION = "2026-01-11 vSmallSample+LegendOutside+x3SD+rwidth0.5"

# ----------------------------
# Helpers
# ----------------------------
def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Normal PDF with a small guard against sd=0."""
    sd = max(float(sd), 1e-12)
    return (1.0 / (sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)


def normal_pdf_max(sd: float) -> float:
    """Peak value of Normal(., sd) PDF."""
    sd = max(float(sd), 1e-12)
    return 1.0 / (sd * np.sqrt(2.0 * np.pi))


def compute_fixed_hist_ylim(
    *,
    S: int,
    x_min: float,
    x_max: float,
    bins: int,
    sigma: float,
    theory_sd: float,
    safety: float = 1.10,
    floor: int = 10,
) -> tuple[int, float]:
    """
    Compute a constant y-axis upper bound for a COUNTS histogram that will not clip
    as samples accumulate to S. Also returns bin_width.

    Uses the maximum possible peak height among:
    - population Normal(μ, σ)
    - sampling-mean Normal(μ, σ/√n)

    Expected peak counts at completion:
        S * bin_width * max_pdf
    """
    bins = max(int(bins), 1)
    bin_width = (float(x_max) - float(x_min)) / bins

    peak_pop = S * bin_width * normal_pdf_max(sigma)
    peak_mean = S * bin_width * normal_pdf_max(theory_sd)

    y_max = int(np.ceil(safety * max(peak_pop, peak_mean, float(floor))))
    y_max = max(y_max, floor)
    return y_max, bin_width


def scale_pdf_to_counts(pdf: np.ndarray, *, completed: int, bin_width: float) -> np.ndarray:
    """Convert a PDF curve to an expected COUNTS curve for 'completed' sample means."""
    completed = max(int(completed), 0)
    return pdf * (completed * float(bin_width))


def add_non_overlapping_legend(ax, fig):
    """Put a small legend outside the axes (right side) so it doesn't cover the plot."""
    fig.subplots_adjust(right=0.78)  # reserve space for legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.4,
    )


# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Sampling Distribution of the Mean", layout="centered")
st.title("Sampling Distribution of the Mean (Small-sample lab scenarios)")

with st.sidebar:
    st.header("Parameters")

    mu = st.number_input("Population mean (μ)", value=0.0, step=0.5)
    sigma = st.number_input("Population SD (σ)", value=1.0, min_value=0.0001, step=0.5)

    # Biomedical-lab-friendly ranges
    n = st.slider("Sample size (n)", min_value=1, max_value=20, value=5)
    S = st.slider("Number of samples (S)", min_value=1, max_value=50, value=20)

    st.divider()

    plot_style = st.radio(
        "Plot style",
        ["Overlay sample means on population", "Histogram of sample means (counts)"],
        index=1,  # default histogram
    )

    bins = st.slider("Histogram bins", min_value=5, max_value=30, value=12)

    st.caption(f"App version: {APP_VERSION}")

    speed = st.slider(
        "Animation speed (sample means per update)",
        min_value=1,
        max_value=4,
        value=1
    )

    delay_ms = st.selectbox(
        "Delay per update (ms)",
        options=[250, 500, 1000, 2000],
        index=1  # default 500
    )

    st.divider()

    show_theory = st.checkbox("Overlay theoretical sampling distribution (μ, σ/√n)", value=True)

    st.subheader("Histogram extras")
    show_population_curve = st.checkbox("Overlay population curve (fixed reference)", value=True)

    st.divider()

    st.subheader("Teaching overlays")
    show_mu_line = st.checkbox("Show μ and mean-of-means lines", value=False)
    show_spread_lines = st.checkbox("Show ±1 SD guides (σ and σ/√n)", value=False)
    show_stats_box = st.checkbox("Show stats box", value=False)

    st.divider()
    st.subheader("Convergence plots")
    show_convergence = st.checkbox("Show live convergence plots", value=False)

start = st.button("Start / Restart Simulation")

plot_area = st.empty()
stats_area = st.empty()
conv_mean_area = st.empty()
conv_sd_area = st.empty()

if not start:
    st.info("Set parameters in the sidebar, then click **Start / Restart Simulation**.")
    st.stop()


# ----------------------------
# Simulation state
# ----------------------------
means: list[float] = []

# History for convergence plots
steps_history: list[int] = []
mean_history: list[float] = []
sd_history: list[float] = []

theory_sd = float(sigma) / np.sqrt(float(n))

# Tightened x-range: μ ± 3σ (per your request)
x_min = float(mu) - 3.0 * float(sigma)
x_max = float(mu) + 3.0 * float(sigma)

# Stable x-grid for smooth curves
xs = np.linspace(x_min, x_max, 600)
pop_pdf = normal_pdf(xs, mu, sigma)
smean_pdf = normal_pdf(xs, mu, theory_sd)

# Overlay-mode y-range (PDF scale). Keep a little headroom.
y_max_pdf = 1.10 * float(max(pop_pdf.max(), smean_pdf.max()))

# Histogram-mode y-range (counts). Now that legend is outside, use 1.10 safety only.
y_max_counts_const, bin_width = compute_fixed_hist_ylim(
    S=S,
    x_min=x_min,
    x_max=x_max,
    bins=bins,
    sigma=sigma,
    theory_sd=theory_sd,
    safety=1.10,
    floor=10,
)

# Convergence plot fixed axes
span_mean = max(1.5 * float(sigma), 4.0 * float(theory_sd), 1e-6)
mean_y_min = float(mu) - span_mean
mean_y_max = float(mu) + span_mean

sd_y_min = 0.0
sd_y_max = 1.10 * max(float(sigma), 3.0 * float(theory_sd), 1e-6)

# Population-curve height adjustment:
# Scale population curve (already "fixed" at S) so its PEAK matches the theory-mean PEAK at S.
# This keeps the population curve visually comparable without dominating.
peak_ratio_pop_to_mean_at_S = normal_pdf_max(theory_sd) / normal_pdf_max(sigma)  # constant factor


# ----------------------------
# Main loop
# ----------------------------
for k in range(0, S, speed):
    batch_size = min(speed, S - k)

    # Draw batch_size samples of size n from Normal(μ, σ)
    samples = np.random.normal(loc=mu, scale=sigma, size=(batch_size, n))
    batch_means = samples.mean(axis=1)
    means.extend(batch_means.tolist())

    means_np = np.array(means, dtype=float)
    completed = int(means_np.size)

    # Empirical stats (so far)
    emp_mean = float(means_np.mean()) if completed >= 1 else float("nan")
    emp_sd = float(means_np.std(ddof=1)) if completed >= 2 else float("nan")

    # Record convergence history at each update
    steps_history.append(completed)
    mean_history.append(emp_mean)
    sd_history.append(emp_sd if completed >= 2 else np.nan)

    # ----------------------------
    # Main plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if plot_style == "Overlay sample means on population":
        ax.plot(xs, pop_pdf, linewidth=2, label="Population: N(μ, σ)")

        if show_theory:
            ax.plot(xs, smean_pdf, linewidth=2, label="Theory for mean: N(μ, σ/√n)")

        if completed > 0:
            jitter = np.random.uniform(0.0, 0.02 * y_max_pdf, size=completed)
            ax.scatter(means_np, jitter, s=18, alpha=0.35)

        ax.set_ylabel("PDF (curves) + sample means (dots)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max_pdf)

    else:
        # Histogram in COUNTS with gaps between bars:
        # rwidth=0.5 makes each bar ~50% of the bin width.
        ax.hist(
            means_np,
            bins=bins,
            range=(x_min, x_max),
            density=False,
            alpha=0.75,
            rwidth=0.5,
            label="Sample means (counts)",
        )
        ax.set_ylabel("Count")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max_counts_const)

        # Sampling-mean theory curve scaled to current completed counts
        if completed > 0 and show_theory:
            ax.plot(
                xs,
                scale_pdf_to_counts(smean_pdf, completed=completed, bin_width=bin_width),
                linewidth=2,
                label="Theory for mean (scaled to counts)",
            )

        # Population curve as FIXED reference:
        # - scaled to final S counts
        # - adjusted so its PEAK matches the theory-mean peak at S
        if show_population_curve:
            pop_fixed = scale_pdf_to_counts(pop_pdf, completed=S, bin_width=bin_width) * peak_ratio_pop_to_mean_at_S
            ax.plot(
                xs,
                pop_fixed,
                linewidth=2,
                linestyle="--",
                alpha=0.6,
                label="Population (fixed ref; peak matched)",
            )

    ax.set_xlabel("Value")
    ax.set_title(f"Sampling distribution building… ({completed} / {S} samples)")

    # ----------------------------
    # Teaching overlays (main plot)
    # ----------------------------
    if show_mu_line:
        ax.axvline(mu, linestyle="--", linewidth=2, alpha=0.9, label="μ (population mean)")
        if completed > 0:
            ax.axvline(emp_mean, linestyle="-", linewidth=2, alpha=0.9, label="Mean of sample means")

    if show_spread_lines:
        ax.axvline(mu - sigma, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(mu + sigma, linestyle="--", linewidth=1, alpha=0.5)

        ax.axvline(mu - theory_sd, linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(mu + theory_sd, linestyle="--", linewidth=1, alpha=0.7)

    if show_stats_box:
        box_text = (
            f"Population:  μ={mu:.3g},  σ={sigma:.3g}\n"
            f"Sampling:    n={n},  S={S},  completed={completed}\n"
            f"Theory SD of mean:  σ/√n={theory_sd:.3g}\n"
        )
        if completed >= 1:
            box_text += f"Empirical mean of means: {emp_mean:.3g}\n"
        if completed >= 2:
            box_text += f"Empirical SD of means:   {emp_sd:.3g}\n"

        ax.text(
            0.02,
            0.98,
            box_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", alpha=0.15),
        )

    add_non_overlapping_legend(ax, fig)
    plot_area.pyplot(fig)
    plt.close(fig)

    # ----------------------------
    # Convergence plots (live)
    # ----------------------------
    if show_convergence:
        fig2, ax2 = plt.subplots(figsize=(7.2, 3.6))
        ax2.plot(steps_history, mean_history, linewidth=2, label="Empirical mean of sample means")
        ax2.axhline(mu, linestyle="--", linewidth=2, alpha=0.9, label="μ (target)")
        ax2.set_xlim(0, S)
        ax2.set_ylim(mean_y_min, mean_y_max)
        ax2.set_xlabel("Samples completed")
        ax2.set_ylabel("Mean of sample means")
        ax2.set_title("Convergence: mean of sample means → μ")
        add_non_overlapping_legend(ax2, fig2)
        conv_mean_area.pyplot(fig2)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7.2, 3.6))
        ax3.plot(steps_history, sd_history, linewidth=2, label="Empirical SD of sample means")
        ax3.axhline(theory_sd, linestyle="--", linewidth=2, alpha=0.9, label="σ/√n (target)")
        ax3.set_xlim(0, S)
        ax3.set_ylim(sd_y_min, sd_y_max)
        ax3.set_xlabel("Samples completed")
        ax3.set_ylabel("SD of sample means")
        ax3.set_title("Convergence: SD of sample means → σ/√n")
        add_non_overlapping_legend(ax3, fig3)
        conv_sd_area.pyplot(fig3)
        plt.close(fig3)

    # ----------------------------
    # Running stats panel
    # ----------------------------
    stats_md = (
        f"**Running stats**\n\n"
        f"- Samples completed: **{completed} / {S}**\n"
        f"- Mean of sample means: **{emp_mean:.6f}** (target μ = {mu})\n"
    )
    if completed >= 2:
        stats_md += f"- SD of sample means: **{emp_sd:.6f}** (target σ/√n = {theory_sd:.6f})\n"
    else:
        stats_md += f"- SD of sample means: **(need ≥ 2 samples)** (target σ/√n = {theory_sd:.6f})\n"

    stats_area.markdown(stats_md)

    time.sleep(int(delay_ms) / 1000.0)

st.success("Done!")
