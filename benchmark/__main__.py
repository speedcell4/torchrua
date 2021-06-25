from aku import Aku

from benchmark.packing import pack_sequence, pack_padded_sequence
from benchmark.padding import pad_sequence, pad_packed_sequence
from benchmark.reduction import tree_reduce, reduce_catted_sequences

app = Aku()

app.option(pack_sequence)
app.option(pack_padded_sequence)
app.option(pad_sequence)
app.option(pad_packed_sequence)
app.option(reduce_catted_sequences)
app.option(tree_reduce)

app.run()
