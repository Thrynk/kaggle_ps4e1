import click

from preprocess import Preprocess
from train import Train

@click.group()
def cli():
    pass

@cli.command()
def preprocess() -> None:
    """Preprocess input data before training.
    """
    p = Preprocess()
    p.main()

@cli.command()
def train() -> None:
    """Train model.
    """
    t = Train()
    t.main()

@cli.command()
def transform() -> None:
    """Transform pipeline to preprocess data and then predict.
    """
    # t = Train()
    # t.main()

if __name__ == "__main__":
    cli()