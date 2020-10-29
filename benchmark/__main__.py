from aku import App

from benchmark.cat_packed_sequence import cat_pack
from benchmark.pack_padded_sequence import pack_padded
from benchmark.pad_packed_sequence import pad_packed
from benchmark.reverse_packed_sequence import reverse_pack
from benchmark.roll_packed_sequence import roll_pack

app = App()

app.register(reverse_pack)
app.register(roll_pack)
app.register(cat_pack)
app.register(pack_padded)
app.register(pad_packed)

app.run()
