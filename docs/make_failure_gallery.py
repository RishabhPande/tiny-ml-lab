"""Draw the failure modes the `test_breaks` tests prove, using the repo's own code.

Each panel reruns an algorithm from algorithms/*/impl.py on the same synthetic
data as its `test_breaks` test. Nothing here is a mock-up.
Needs matplotlib (plotting only; the algorithms stay numpy only):
    pip install matplotlib
    python docs/make_failure_gallery.py     # writes docs/failure-modes.png
"""

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, RED, GREEN, GREY = "#2e86ab", "#e4572e", "#59a14f", "#555555"


def load(name):
    path = os.path.join(HERE, "..", "algorithms", name, "impl.py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kmeans_panel(ax):
    KMeans = load("kmeans").KMeans
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    inner = np.column_stack([np.cos(theta[:100]), np.sin(theta[:100])])
    inner += rng.normal(0, 0.1, inner.shape)
    outer = 4.0 * np.column_stack([np.cos(theta[100:]), np.sin(theta[100:])])
    outer += rng.normal(0, 0.1, outer.shape)
    X = np.vstack([inner, outer])
    km = KMeans(n_clusters=2, seed=0).fit(X)
    ax.scatter(X[:, 0], X[:, 1], c=[BLUE if l == 0 else RED for l in km.labels_], s=10)
    ax.set_aspect("equal")
    ax.set_title("K-Means on two rings\nclusters split by angle, not by ring", fontsize=10)


def pca_panel(ax):
    PCA = load("pca").PCA
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 300)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += rng.normal(0, 0.05, X.shape)
    pca = PCA(n_components=1).fit(X)
    ax.scatter(X[:, 0], X[:, 1], s=10, color=BLUE)
    v = pca.components_[0]
    ax.plot([-1.4 * v[0], 1.4 * v[0]], [-1.4 * v[1], 1.4 * v[1]], color=RED, lw=2)
    ax.set_aspect("equal")
    ax.set_title("PCA on a circle\nfirst component keeps only "
                 f"{pca.explained_variance_ratio_[0]:.0%} of the variance", fontsize=10)


def logistic_panel(ax):
    LR = load("logistic-regression").LogisticRegression
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 3.0]])
    labels = np.array([0, 1, 1, 0])
    X = np.vstack([c + rng.normal(0, 0.4, (50, 2)) for c in centers])
    y = np.repeat(labels, 50)
    m = LR(lr=0.5, n_iter=2000).fit(X, y)
    acc = float(np.mean(m.predict(X) == y))
    ax.scatter(X[:, 0], X[:, 1], c=[BLUE if l == 0 else RED for l in y], s=10)
    gx, gy = np.meshgrid(np.linspace(-1, 4, 200), np.linspace(-1, 4, 200))
    p = m.predict_proba(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contour(gx, gy, p, levels=[0.5], colors=[GREY], linewidths=2)
    ax.set_aspect("equal")
    ax.set_title(f"Logistic regression on XOR\nstraight boundary, accuracy {acc:.0%}", fontsize=10)


def linreg_panel(ax):
    lr = load("linear-regression")
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, 100)
    y = 2.0 * X + 1.0 + rng.normal(0, 1.0, 100)
    clean = lr.fit(X, y)
    Xb, yb = np.append(X, 10.0), np.append(y, 1000.0)
    bad = lr.fit(Xb, yb)
    xs = np.linspace(0, 10, 2)
    ax.scatter(X, y, s=10, color=BLUE)
    ax.scatter([10.0], [1000.0], s=40, color=RED)
    ax.plot(xs, lr.predict(xs, clean), color=GREEN, lw=2, label=f"clean fit, slope {clean[1]:.1f}")
    ax.plot(xs, lr.predict(xs, bad), color=RED, lw=2, label=f"with 1 outlier, slope {bad[1]:.1f}")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Least squares with one outlier\nslope dragged by a single point", fontsize=10)


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for fn, ax in zip((kmeans_panel, pca_panel, logistic_panel, linreg_panel), axes):
        fn(ax)
    fig.text(0.01, 0.01, "Same synthetic data and seeds as each algorithm's test_breaks test, "
             "run through algorithms/*/impl.py. Reproduce: python docs/make_failure_gallery.py",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3.0)
    fig.savefig(os.path.join(HERE, "failure-modes.png"), dpi=150)


if __name__ == "__main__":
    main()
