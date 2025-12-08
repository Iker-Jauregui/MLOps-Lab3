"""
Main CLI or app entry point
"""

import click
from mylib.classifier import predict


# We create a group of commands
@click.group()
def cli():
    """Main CLI to perform image operations."""


# ============================================================================
# INFERENCE GROUP - Image inference operations
# ============================================================================
@cli.group()
def inference():
    """Commands for image inference operations."""


# We create a command, named predict, associated with the previous group
@inference.command("predict")
@click.argument("image-path", type=str)
def predict_cli(image_path):
    """Predict image class.

    Example:
        uv run python -m cli.cli inference predict 'sample.jpg'
    """
    try:
        result = predict(image_path)
        click.echo(click.style(f"Predicted class: {result}", fg="green"))
    except (FileNotFoundError, IOError, ValueError) as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)


# Main entry point
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
