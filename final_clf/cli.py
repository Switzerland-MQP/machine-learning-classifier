import click
import plaintext_converter
import run


@click.group()
def cli():
    pass


@cli.command()
@click.argument("source")
@click.argument("dest")
def ocr(source, dest):
    click.echo("Ocr")
    plaintext_converter.convert_directory(source, dest)


@cli.command()
@click.argument("filepath")
def run_model(filepath):
    print("Run Model")
    run.run_model(filepath)


if __name__ == "__main__":
    cli()

