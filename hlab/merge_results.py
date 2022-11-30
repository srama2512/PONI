import glob
import json

import click
import numpy as np


METRICS = ["success", "spl", "distance_to_goal"]
PM = "\u00B1"  # Plus-minus symbol


@click.command()
@click.option("--path_format", default="./result.log")
def get_results(path_format: str):
    paths = glob.glob(path_format)
    print(f"# Paths: {len(paths)}")
    metric_values = {m: [] for m in METRICS}
    for path in paths:
        try:
            path_metrics = get_path_results(path)
            for m in METRICS:
                metric_values[m] += path_metrics[m]
        except Exception as e:
            print(f"Failed path: {path}")
            print(e)
            continue
    results = []
    for m in METRICS:
        results.append("{:.1f}".format(np.mean(metric_values[m]).item()))
    print(",".join(results))


def get_mean_stddev(results):
    mean = np.mean(results).item()
    stddev = np.std(results).item()
    # return f'{mean:.3f} {PM} {stddev:.3f}'
    return f"{mean:.1f}"


def get_path_results(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    metric_values = {m: [] for m in METRICS}
    for k, v in data.items():
        for metric, value in v.items():
            if metric not in METRICS:
                continue
            if metric in ["success", "spl", "softspl"]:
                value *= 100.0
            metric_values[metric].append(value)
    return metric_values


if __name__ == "__main__":
    get_results()
