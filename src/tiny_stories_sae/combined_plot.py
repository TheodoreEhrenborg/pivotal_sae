import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import seaborn as sns
from beartype import beartype


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--response_jsons", type=Path, required=True, nargs="+")
    parser.add_argument("--labels", type=str, required=True, nargs="+")
    parser.add_argument("--output_file", type=str, default="/results/out.png")
    parser.add_argument("--xlabel", type=str)
    parser.add_argument("--title", type=str)
    return parser


@beartype
def get_dataframe(path: Path, label: str) -> pd.DataFrame:
    data = json.load(path.open())
    assert data["model"] == "gpt-4o-2024-08-06", "Don't use results from GPT-4o mini"

    frame = pd.DataFrame(data["responses"])
    frame["Legend"] = label
    return frame


@beartype
def main(args: Namespace) -> None:
    sns.set_theme()
    frames = [
        get_dataframe(path, label)
        for path, label in zip(args.response_jsons, args.labels, strict=True)
    ]
    df = pd.concat(frames)

    seaborn_plot = sns.countplot(df, x="clearness", hue="Legend")
    if args.xlabel is not None:
        seaborn_plot.set(xlabel=args.xlabel)
    if args.title is not None:
        seaborn_plot.set(title=args.title)
    fig = seaborn_plot.get_figure()
    print(f"Writing to {args.output_file}")
    fig.savefig(args.output_file)


if __name__ == "__main__":
    main(make_parser().parse_args())
