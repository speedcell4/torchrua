from aku import App

from benchmark.cat_packed_sequence import cat_packed_sequence_fn
from benchmark.pack_padded_sequence import pack_padded_sequence_fn
from benchmark.pad_packed_sequence import pad_packed_sequence_fn
from benchmark.reverse_packed_sequence import reverse_pack_fn

app = App()

app.register(reverse_pack_fn)
app.register(cat_packed_sequence_fn)
app.register(pack_padded_sequence_fn)
app.register(pad_packed_sequence_fn)

app.run()
