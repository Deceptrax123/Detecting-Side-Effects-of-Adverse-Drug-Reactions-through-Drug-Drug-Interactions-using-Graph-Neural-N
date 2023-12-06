# Note to Repository Contributors
## Follow the steps below
- Run ```git clone git@github.com:Deceptrax123/Drug-Interaction-Using-GNNs.git```
- Create a Virtual environment and activate it
- Run ```pip install -r requirements.txt```
- Create a new branch for every new model such as GAT, GCN, SAGE etc. For instance,  run ```git checkout -b GCN``` to create and run a Graph Convolutional Model
- Commit and Push all changes to that specific branch only. I will review and merge if no further changes are needed.

## Dataset
You may download all the Graph Tensors with their folds over <a href="https://drive.google.com/file/d/1zQx6fv1Y_BEb11zE77OfM-9M4i-gTjXJ/view?usp=sharing">here</a>
  

## Note
- Before contributing, run the following command: ```export PYTHONPATH="/path/to/your/project"```
- Follow the same directoy structure as follws: 

```  
├── Dataset
│   ├── Molecule_dataset.py
├── Metrics
│   ├── metrics.py
├── GAT
│   ├── train.py
│   ├── model.py
├── GCN
│   ├── train.py
│   ├── model.py
├── SAGE
│   ├── ....
│   ├── ....
├── requirements.txt
├── contributing.md
└── .gitignore
```