# -*- coding: utf-8 -*-

"""Console script for keras_trans."""
import tensorflow
import sys
import click
from pprint import pprint

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from seq2seq.model import FCholletSeq2Seq


@click.command()
def main(args=None):
    """Console script for keras_trans."""
    click.echo("Replace this message by putting your code into "
               "keras_trans.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    path = '/Users/Olamilekan/Desktop/Machine Learning/OpenSource/lms/data/fra-eng/fra.txt'
    a = FCholletSeq2Seq(data_path=path, fformat='file', nlevel='char')
    print("-----------Model Details----------")

    pprint(a.params)

    print("---------------------")

    a.fit()
    sys.exit(main())  # pragma: no cover
