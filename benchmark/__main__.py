from aku import App

from benchmark.reverse_packed_sequence import reverse_pack_fn
from benchmark.cat_packed_sequence import cat_packed_sequence_fn

app = App()

app.register(reverse_pack_fn)
app.register(cat_packed_sequence_fn)

app.run()
