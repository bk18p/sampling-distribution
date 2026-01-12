import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


APP_VERSION = "2026-01-11 true-pop-pdf v1"


# ----------------------------
# PDFs / utilities
# ----------------------------
def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-12)
    return (1.0 / (sd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / sd) ** 2)


def normal_pdf_max(sd: float) -> float:
    sd = max(float(sd), 1e-12)
    return 1.0 / (sd * np.sqrt(2.0 * np.pi))


def lognormal_params_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    """
    Return (m, s) such that if Y ~ LogNormal(m, s) then E[Y]=mean and SD[Y]=sd.
    """
    mean = max(float(mean), 1e-12)
    sd = max(float(sd), 1e-12)
    var = sd * sd
    s2 = np.log(1.0 + var / (mean * mean))
    s = float(np.sqrt(s2))
    m = float(np.log(mean) - 0.5 * s2)
    return m, s


def lognormal_pdf(y: np.ndarray, m: float, s: float) -> np.ndarray:
    """
    PDF of LogNormal(m, s) evaluated at y (y must be > 0; returns 0 otherwise).
    """
    y = np.asarray(y, dtype=float)
    s = max(float(s), 1e-12)
    out = np.zeros_like(y)
    mask = y > 0
    yy = y[mask]
    out[mask] = (1.0 / (yy * s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((np.log(yy) - m) / s) ** 2)
    return out


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
    Fixed y-axis upper bound for a COUNTS histogram that won't clip by final S.
    Returns (y_max_counts_const, bin_width).

    Uses a conservative expected-peak calculation based on Normal peaks.
    """
    bins = max(int(bins), 1)
    bin_width = (float(x_max) - float(x_min)) / bins

    peak_pop = S * bin_width * normal_pdf_max(sigma)
    peak_mean = S * bin_width * normal_pdf_max(theory_sd)

    y_max = int(np.ceil(safety * max(peak_pop, peak_mean, float(floor))))
    return max(y_max, floor), bin_width


def scale_pdf_to_counts(pdf: np.ndarray, *, completed: int, bin_width: float) -> np.ndarray:
    completed = max(int(completed), 0)
    return pdf * (completed * float(bin_width))


def add_non_overlapping_legend(ax, fig):
    fig.subplots_adjust(right=0.78)
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
# Distribution: sampling + true PDF/PMF
# ----------------------------
def dist_internal_params(dist: str, mu: float, sigma: float) -> dict:
    """
    Precompute any deterministic parameters for the chosen distribution
    so mean=mu and sd=sigma exactly (up to floating error).
    """
    mu = float(mu)
    sigma = max(float(sigma), 1e-12)

    if dist == "Normal":
        return {"mu": mu, "sigma": sigma}

    if dist == "Uniform":
        half_range = sigma * np.sqrt(3.0)
        a = mu - half_range
        b = mu + half_range
        return {"a": a, "b": b}

    if dist == "Shifted Exponential":
        # X = (Exp(scale=sigma) - sigma) + mu
        # support: [mu - sigma, +inf)
        return {"mu": mu, "sigma": sigma}

    if dist == "Two-point (±1 SD)":
        return {"x1": mu - sigma, "x2": mu + sigma}

    if dist == "Bimodal mixture (two normals)":
        # 50/50 mixture of N(mu-d, s0) and N(mu+d, s0) with s0^2 + d^2 = sigma^2
        s0 = 0.5 * sigma
        d = np.sqrt(max(sigma**2 - s0**2, 0.0))
        return {"mu": mu, "s0": s0, "d": d}

    if dist == "Lognormal (shifted)":
        # Construct Y ~ LogNormal(m,s) with mean mY and sd sigma,
        # then X = Y + c with c = mu - mY so mean becomes mu; SD unchanged.
        mY = max(0.25 * sigma, 1e-6)  # positive base mean (tunable)
        m, s = lognormal_params_from_mean_sd(mY, sigma)
        c = mu - mY
        return {"mY": mY, "m": m, "s": s, "c": c}

    raise ValueError(f"Unknown distribution: {dist}")


def sample_population(dist: str, params: dict, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """
    Sample from chosen distribution with target mean & SD.
    Returns array with shape == size.
    """
    if dist == "Normal":
        return rng.normal(loc=params["mu"], scale=params["sigma"], size=size)

    if dist == "Uniform":
        return rng.uniform(low=params["a"], high=params["b"], size=size)

    if dist == "Shifted Exponential":
        exp_part = rng.exponential(scale=params["sigma"], size=size)
        return (exp_part - params["sigma"]) + params["mu"]

    if dist == "Two-point (±1 SD)":
        signs = rng.choice([-1.0, 1.0], size=size)
        # x1=mu-sigma, x2=mu+sigma equivalently:
        mu_mid = 0.5 * (params["x1"] + params["x2"])
        sigma = 0.5 * (params["x2"] - params["x1"])
        return mu_mid + sigma * signs

    if dist == "Bimodal mixture (two normals)":
        mu = params["mu"]
        s0 = params["s0"]
        d = params["d"]
        choose = rng.random(size=size) < 0.5
        out = np.empty(size, dtype=float)
        out[choose] = rng.normal(loc=mu - d, scale=s0, size=out[choose].shape)
        out[~choose] = rng.normal(loc=mu + d, scale=s0, size=out[~choose].shape)
        return out

    if dist == "Lognormal (shifted)":
        # X = Y + c
        Y = rng.lognormal(mean=params["m"], sigma=params["s"], size=size)
        return Y + params["c"]

    raise ValueError(f"Unknown distribution: {dist}")


def population_pdf_or_spikes(dist: str, params: dict, xs: np.ndarray) -> dict:
    """
    Return either:
      - {"type":"pdf", "y": pdf(xs), "label": "..."}
    or
      - {"type":"spikes", "x": [..], "p": [..], "label":"..."} for discrete distributions
    """
    xs = np.asarray(xs, dtype=float)

    if dist == "Normal":
        mu, sigma = params["mu"], params["sigma"]
        return {"type": "pdf", "y": normal_pdf(xs, mu, sigma), "label": "Population PDF"}

    if dist == "Uniform":
        a, b = params["a"], params["b"]
        y = np.zeros_like(xs)
        mask = (xs >= a) & (xs <= b)
        y[mask] = 1.0 / max(b - a, 1e-12)
        return {"type": "pdf", "y": y, "label": "Population PDF (Uniform)"}

    if dist == "Shifted Exponential":
        mu, sigma = params["mu"], params["sigma"]
        # support starts at (mu - sigma)
        x0 = mu - sigma
        y = np.zeros_like(xs)
        mask = xs >= x0
        y[mask] = (1.0 / sigma) * np.exp(-(xs[mask] - x0) / sigma)
        return {"type": "pdf", "y": y, "label": "Population PDF (Shifted Exponential)"}

    if dist == "Two-point (±1 SD)":
        x1, x2 = params["x1"], params["x2"]
        # PMF: P(x1)=P(x2)=0.5
        return {"type": "spikes", "x": [x1, x2], "p": [0.5, 0.5], "label": "Population PMF (Two-point)"}

    if dist == "Bimodal mixture (two normals)":
        mu, s0, d = params["mu"], params["s0"], params["d"]
        y = 0.5 * normal_pdf(xs, mu - d, s0) + 0.5 * normal_pdf(xs, mu + d, s0)
        return {"type": "pdf", "y": y, "label": "Population PDF (Bimodal mixture)"}

    if dist == "Lognormal (shifted)":
        # X = Y + c, so f_X(x) = f_Y(x-c) for x>c else 0
        c, m, s = params["c"], params["m"], params["s"]
        y = lognormal_pdf(xs - c, m, s)
        return {"type": "pdf", "y": y, "label": "Population PDF (Shifted Lognormal)"}

    raise ValueError(f"Unknown distribution: {dist}")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sampling Distribution of the Mean", layout="centered")
st.title("Sampling Distribution of the Mean (Small-sample lab scenarios)")

with st.sidebar:
    st.header("Parameters")

    mu = st.number_input("Target mean (μ)", value=0.0, step=0.5)
    sigma = st.number_input("Target SD (σ)", value=1.0, min_value=0.0001, step=0.5)

    dist = st.selectbox(
        "Underlying distribution (true PDF shown)",
        [
            "Normal",
            "Uniform",
            "Shifted Exponential",
            "Two-point (±1 SD)",
            "Bimodal mixture (two normals)",
            "Lognormal (shifted)",
        ],
        index=0,
    )

    # Small-sample lab defaults / ranges
    n = st.slider("Sample size (n)", min_value=1, max_value=20, value=5)
    S = st.slider("Number of samples (S)", min_value=1, max_value=50, value=20)

    st.divider()

    plot_style = st.radio(
        "Plot style",
        ["Overlay sample means on population", "Histogram of sample means (counts)"],
        index=1,  # default histogram
    )

    bins = st.slider("Histogram bins", min_value=5, max_value=30, value=12)

    speed = st.slider("Animation speed (sample means per update)", min_value=1, max_value=4, value=1)

    delay_ms = st.selectbox(
        "Delay per update (ms)",
        options=[250, 500, 1000, 2000],
        index=1,
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

    st.divider()
    st.caption(f"App version: {APP_VERSION}")

start = st.button("Start / Restart Simulation")

plot_area = st.empty()
stats_area = st.empty()
conv_mean_area = st.empty()
conv_sd_area = st.empty()

if not start:
    st.info("Set parameters in the sidebar, then click **Start / Restart Simulation**.")
    st.stop()


# ----------------------------
# Precompute fixed pieces
# ----------------------------
rng = np.random.default_rng()

params = dist_internal_params(dist, mu, sigma)
theory_sd = float(sigma) / np.sqrt(float(n))

# x-range: μ ± 3σ
x_min = float(mu) - 3.0 * float(sigma)
x_max = float(mu) + 3.0 * float(sigma)
xs = np.linspace(x_min, x_max, 600)

pop_info = population_pdf_or_spikes(dist, params, xs)
smean_pdf = normal_pdf(xs, mu, theory_sd)

# For overlay mode y-limit:
# - if pdf: use max(pop_pdf, theory_pdf)
# - if spikes: use theory peak (spikes drawn relative to that)
if pop_info["type"] == "pdf":
    pop_pdf = pop_info["y"]
    y_max_pdf = 1.10 * float(max(pop_pdf.max(), smean_pdf.max()))
else:
    pop_pdf = None
    y_max_pdf = 1.10 * float(smean_pdf.max())

# Histogram y-limit (counts)
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

# Peak-match factor for population curve in histogram mode:
# Make max(pop_curve_scaled_to_S) equal max(theory_curve_scaled_to_S)
theory_peak_counts_at_S = float((smean_pdf * (S * bin_width)).max())

if pop_info["type"] == "pdf":
    pop_peak_pdf = float(np.max(pop_info["y"]))
    pop_peak_counts_at_S = float((pop_info["y"] * (S * bin_width)).max())
    peak_match_factor = (theory_peak_counts_at_S / pop_peak_counts_at_S) if pop_peak_counts_at_S > 0 else 1.0
else:
    # spikes will be drawn with heights tied directly to theory peak
    peak_match_factor = 1.0

# Convergence histories
means: list[float] = []
steps_history: list[int] = []
mean_history: list[float] = []
sd_history: list[float] = []

span_mean = max(1.5 * float(sigma), 4.0 * float(theory_sd), 1e-6)
mean_y_min = float(mu) - span_mean
mean_y_max = float(mu) + span_mean
sd_y_min = 0.0
sd_y_max = 1.10 * max(float(sigma), 3.0 * float(theory_sd), 1e-6)


# ----------------------------
# Main loop
# ----------------------------
for k in range(0, S, speed):
    batch_size = min(speed, S - k)

    samples = sample_population(dist, params, (batch_size, n), rng)
    batch_means = samples.mean(axis=1)
    means.extend(batch_means.tolist())

    means_np = np.array(means, dtype=float)
    completed = int(means_np.size)

    emp_mean = float(means_np.mean()) if completed >= 1 else float("nan")
    emp_sd = float(means_np.std(ddof=1)) if completed >= 2 else float("nan")

    steps_history.append(completed)
    mean_history.append(emp_mean)
    sd_history.append(emp_sd if completed >= 2 else np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    if plot_style == "Overlay sample means on population":
        # True population PDF (or PMF spikes)
        if pop_info["type"] == "pdf":
            ax.plot(xs, pop_info["y"], linewidth=2, label=pop_info["label"])
        else:
            # Draw spikes up to ~80% of plot height for visibility
            spike_height = 0.8 * y_max_pdf
            ax.vlines(pop_info["x"], 0, spike_height, linewidth=3, label=pop_info["label"])

        if show_theory:
            ax.plot(xs, smean_pdf, linewidth=2, label="Theory for mean: Normal(μ, σ/√n)")

        if completed > 0:
            jitter = np.random.uniform(0.0, 0.02 * y_max_pdf, size=completed)
            ax.scatter(means_np, jitter, s=18, alpha=0.35, label="Sample means")

        ax.set_ylabel("Density (PDF) + sample means (dots)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max_pdf)

    else:
        # Histogram counts with gaps
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

        # Theory curve for mean scaled to counts so far
        if completed > 0 and show_theory:
            ax.plot(
                xs,
                scale_pdf_to_counts(smean_pdf, completed=completed, bin_width=bin_width),
                linewidth=2,
                label="Theory for mean (scaled to counts)",
            )

        # True population curve fixed at final S (peak-matched to theory peak at S)
        if show_population_curve:
            if pop_info["type"] == "pdf":
                pop_fixed_counts = scale_pdf_to_counts(pop_info["y"], completed=S, bin_width=bin_width) * peak_match_factor
                ax.plot(
                    xs,
                    pop_fixed_counts,
                    linewidth=2,
                    linestyle="--",
                    alpha=0.6,
                    label="Population (true PDF; fixed at S; peak matched)",
                )
            else:
                # spikes: draw at fixed final height ~ theory peak at S
                spike_height_counts = 0.9 * theory_peak_counts_at_S
                ax.vlines(
                    pop_info["x"],
                    0,
                    spike_height_counts,
                    linewidth=3,
                    linestyles="--",
                    alpha=0.6,
                    label="Population (PMF spikes; fixed; peak matched)",
                )

    ax.set_xlabel("Value")
    ax.set_title(f"Underlying: {dist} | building… ({completed} / {S})")

    # Teaching overlays
    if show_mu_line:
        ax.axvline(mu, linestyle="--", linewidth=2, alpha=0.9, label="μ")
        if completed > 0:
            ax.axvline(emp_mean, linestyle="-", linewidth=2, alpha=0.9, label="Mean of sample means")

    if show_spread_lines:
        ax.axvline(mu - sigma, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(mu + sigma, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(mu - theory_sd, linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(mu + theory_sd, linestyle="--", linewidth=1, alpha=0.7)

    if show_stats_box:
        box_text = (
            f"Underlying: {dist}\n"
            f"Target μ={mu:.3g}, σ={sigma:.3g}\n"
            f"n={n}, S={S}, completed={completed}\n"
            f"Theory SD of mean: σ/√n={theory_sd:.3g}\n"
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

    # Convergence plots
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

    # Running stats
    stats_md = (
        f"**Running stats**\n\n"
        f"- Underlying: **{dist}**\n"
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
