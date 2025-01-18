import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas
import seaborn as sns
from beartype import beartype


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--response_json", type=Path, required=True)
    parser.add_argument("--output_file", type=str, default="/results/out.png")
    parser.add_argument("--xlabel", type=str)
    parser.add_argument("--title", type=str)
    return parser


def main(args: Namespace):
    sns.set_theme()
    data = json.load(args.response_json.open())
    df = pandas.DataFrame(data["responses"])
    seaborn_plot = sns.countplot(df, x="clearness")
    if args.xlabel is not None:
        seaborn_plot.set(xlabel=args.xlabel)
    if args.title is not None:
        seaborn_plot.set(title=args.title)
    fig = seaborn_plot.get_figure()
    print(f"Writing to {args.output_file}")
    fig.savefig(args.output_file)


if __name__ == "__main__":
    main(make_parser().parse_args())
