from aku import Aku

from benchmark.compose import compose_catted_sequences
from benchmark.packing import pack_sequence, pack_padded_sequence
from benchmark.padding import pad_sequence, pad_packed_sequence
from benchmark.reduction import reduce_packed_sequence, reduce_catted_sequence, reduce_padded_sequence

# from benchmark.scatter import scatter_add, scatter_softmax, scatter_logsumexp, scatter_max, scatter_mul

app = Aku()

app.option(pack_sequence)
app.option(pack_padded_sequence)
app.option(pad_sequence)
app.option(pad_packed_sequence)
app.option(compose_catted_sequences)
app.option(reduce_catted_sequence)
app.option(reduce_packed_sequence)
app.option(reduce_padded_sequence)
# app.option(scatter_add)
# app.option(scatter_mul)
# app.option(scatter_max)
# app.option(scatter_logsumexp)
# app.option(scatter_softmax)

app.run()
