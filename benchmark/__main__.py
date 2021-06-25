from aku import Aku

from benchmark.reduction import tree_reduce

app = Aku()

app.option(tree_reduce)

app.run()
