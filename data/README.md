# Data

The project uses the Penn Treebank sample that ships with NLTK:

```python
from nltk.corpus import treebank
treebank.tagged_sents()
```

It contains 3,914 tagged English sentences. The notebook downloads the corpus
automatically on first run (`nltk.download("treebank")`), so no data files need to
be committed to this repository.

The reproducible 80/10/10 split is defined in `src/data.py` with seed 42.
