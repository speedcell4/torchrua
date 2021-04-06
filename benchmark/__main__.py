from aku import Aku

from benchmark.cat_packed_sequence import cat_pack
from benchmark.pack_padded_sequence import pack_padded
from benchmark.pad_packed_sequence import pad_packed
from benchmark.reverse_packed_sequence import reverse_pack
from benchmark.roll_packed_sequence import roll_pack

app = Aku()

app.option(reverse_pack)
app.option(roll_pack)
app.option(cat_pack)
app.option(pack_padded)
app.option(pad_packed)

app.run()
