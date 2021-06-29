from aku import Aku

from benchmark.packing import pack_sequence, pack_padded_sequence, pack_catted_sequences
from benchmark.padding import pad_sequence, pad_packed_sequence
from benchmark.reduction import tree_reduce_packed_sequence

app = Aku()

app.option(pack_sequence)
app.option(pack_padded_sequence)
app.option(pad_sequence)
app.option(pad_packed_sequence)
app.option(pack_catted_sequences)
app.option(tree_reduce_packed_sequence)

app.run()
