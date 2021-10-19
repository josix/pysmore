"""Test Python scripts through invoking pytest and coverage"""
from invoke import task


@task
def pytest(ctx):
    """Run pytest"""
    ctx.run("pytest")


@task
def coverage(ctx):
    """Run pytest with coverage"""
    ctx.run("coverage run -m pytest")
    ctx.run("coverage report -m")


@task(pre=[pytest, coverage], default=True)
def test(ctx):  # pylint: disable=unused-argument
    """Reformat Python scripts through coverage and pytest"""
    return
