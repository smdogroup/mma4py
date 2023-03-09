import pandas as pd
import matplotlib.pyplot as plt
import argparse


def mma4py_plot_log(ax, log_path):
    # Load from history, drop repeated header rows and reset indices
    df = pd.read_fwf(log_path)
    df = df[df.ne(df.columns).any(axis="columns")]
    df.reset_index(drop=True, inplace=True)
    df = df.astype(float)

    # Plot objective using primary axis

    l0 = ax.plot(df["iter"], df["obj"], color="blue", label="obj")

    # Plot KKT error and infeasibility using secondary axis (with log-scale)
    ax2 = ax.twinx()
    l1 = ax2.semilogy(
        df["iter"], df["KKT_l2"], color="orange", alpha=0.8, label="KKT l2"
    )
    l2 = ax2.semilogy(
        df["iter"], df["KKT_linf"], color="orange", alpha=0.5, label="KKT linf"
    )
    l3 = ax2.semilogy(
        df["iter"], df["infeas"], color="purple", alpha=0.5, label="infeas"
    )

    # Manually set legends
    lns = l0 + l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.set_xlabel("iter")
    ax.set_ylabel("obj")
    ax2.set_xlabel("opt/feas criteria")

    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser("Plot values from a mma4py output file")
    p.add_argument(
        "filename", metavar="./mma4py.log", type=str, help="path to mma4py log file"
    )
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    mma4py_plot_log(args.filename)
    plt.show()
