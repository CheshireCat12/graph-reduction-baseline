"""
Retrieve the original graphs from the TUDataset and save them as graphml.
The corresponding classes are saved in a separate file.
"""
import argparse

import numpy as np
from utils import load_graphs_from_TUDataset, save_graphs


def main(args):
    nx_graphs, graph_classes = load_graphs_from_TUDataset(args.root_dataset,
                                                          args.dataset)

    save_graphs(args.folder_results, nx_graphs, graph_classes)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create Baseline Graphs.\n'
                    '1. Retrieve the graph dataset from the TUDataset repo.\n'
                    '2. Retrieve the classes of the graphs.\n'
                    '3. Transform the graphs from the PyG representation to NetworkX.Graph.\n'
                    '4. Save the graphs and the corresponding classes.\n\n'
                    'Result example:\n'
                    '--------\n'
                    'Folder\n'
                    '  |- graph_0.graphml\n'
                    '  |- graph_1.graphml\n'
                    '  |- ....\n'
                    '  |- graph_classes.cxl')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--dataset',
                             type=str,
                             required=True,
                             help='Graph dataset to retrieve'
                                  '(the chosen dataset has to be in the TUDataset repository)')
    args_parser.add_argument('--root_dataset',
                             type=str,
                             default='/tmp/data',
                             help='Root of the TUDataset')

    args_parser.add_argument('--folder_results',
                             type=str,
                             required=True,
                             help='Folder where to save the `graphml` graphs and their corresponding classes.')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    if parse_args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(parse_args)
