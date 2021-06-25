from aku import Aku

from benchmark.packing import pack_sequence, pack_padded_sequence
from benchmark.reduction import tree_reduce

app = Aku()

app.option(pack_sequence)
app.option(pack_padded_sequence)
app.option(tree_reduce)

app.run()
