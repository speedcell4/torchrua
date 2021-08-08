from aku import Aku

from benchmark.attention import scatter_attention
from benchmark.packing import pack_sequence, pack_padded_sequence
from benchmark.padding import pad_sequence, pad_packed_sequence
from benchmark.reduction import reduce_catted_sequences
from benchmark.scatter import scatter_add, scatter_softmax, scatter_logsumexp, scatter_max, scatter_mul
from benchmark.tree_reduction import tree_reduce_catted_sequence
from benchmark.tree_reduction import tree_reduce_packed_sequence, tree_reduce_padded_sequence

app = Aku()

app.option(pack_sequence)
app.option(pack_padded_sequence)
app.option(pad_sequence)
app.option(pad_packed_sequence)
app.option(reduce_catted_sequences)
app.option(tree_reduce_packed_sequence)
app.option(tree_reduce_padded_sequence)
app.option(tree_reduce_catted_sequence)
app.option(scatter_add)
app.option(scatter_mul)
app.option(scatter_max)
app.option(scatter_logsumexp)
app.option(scatter_softmax)
app.option(scatter_attention)

app.run()
