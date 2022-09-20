# Baseline Graphs From TUDataset

This python module is used to retrieve the graph datasets from [TUDataset](https://chrsmrrs.github.io/datasets/).
- First, we use the [Pytorch Geometric]() loader to download the graph datasets.
- Then, we transform the graphs into NetworkX graphs before saving them in `graph_idx.graphml`.

## Install

1. Pytorch needs to be installed before the other dependencies.
The pytorch version (1.12.0) may be important to make the torch-geometric's dependency modules work properly.

    ```
    pip install torch==1.12.0
    ```

2. Install the other dependencies

    ```
    pip install -r requirements.txt
    ```

If a segmentation fault appears try installing by hand (problems during the installation of torch_geometric)

    pip install numpy networkx tqdm torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html


## How to use

### Example

Run the following line to download and generate the Enzyme dataset.

```
python baseline.py --dataset ENZYMES --folder_results ./data_tmp/enzymes
```

Resulting folder

```
├── enzymes
|   ├── graph_0.graphml
|   ├── graph_1.graphml
|   ├── graph_2.graphml
|   ├── graph_3.graphml
|       ....
|   ├── graph_classes.cxl
```

Example of a generated `.graphml` file containing a graph.

```xml
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="https://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://graphml.graphdrawing.org/xmlns ">
   <key id="d0" for="node" attr.name="x" attr.type="string" />
   <graph edgedefault="undirected">
      <node id="0">
         <data key="d0">[1. 0. 0.]</data>
      </node>
      <node id="1">
         <data key="d0">[1. 0. 0.]</data>
      </node>
      ...
      <edge source="0" target="1" />
      <edge source="0" target="13" />
      <edge source="1" target="14" />
      <edge source="1" target="19" />
      ...
   </graph>
</graphml>
```

XML file containing the class for each graph

```xml
<?xml version="1.0" ?>
<GraphCollection>
   <idx_graph_to_classes>
      <element graph_file="gr_0.graphml" class="0"/>
      <element graph_file="gr_1.graphml" class="0"/>
      <element graph_file="gr_2.graphml" class="0"/>
      ...
   </idx_graph_to_classes>
</GraphCollection>
```

Help page if needed

```
usage: baseline.py [-h] --dataset DATASET [--root_dataset ROOT_DATASET] --folder_results FOLDER_RESULTS [-v] {} ...

Create Baseline Graphs.
1. Retrieve the graph dataset from the TUDataset repo.
2. Retrieve the classes of the graphs.
3. Transform the graphs from the PyG representation to NetworkX.Graph.
4. Save the graphs and the corresponding classes.

Result example:
--------
├── Folder_results
|   ├── graph_0.graphml
|   ├── graph_1.graphml
|       ....
|   ├── graph_classes.cxl

positional arguments:
  {}

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Graph dataset to retrieve(the chosen dataset has to be in the TUDataset repository)
  --root_dataset ROOT_DATASET
                        Root of the TUDataset
  --folder_results FOLDER_RESULTS
                        Folder where to save the `graphml` graphs and their corresponding classes.
  -v, --verbose         Activate verbose print

```
