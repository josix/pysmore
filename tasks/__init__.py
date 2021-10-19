"""
Module that collects all tha available invoking commands
"""
from invoke import Collection, task

from tasks import lint, reformat, test


@task
def all_task(ctx):
    """
    Invokes all tasks
    """
    ctx.run("invoke reformat")
    ctx.run("invoke lint")
    ctx.run("invoke test")


namespace = Collection(reformat, lint, test, all_task)
