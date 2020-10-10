from aku import App

from benchmark.reverse_packed_sequence import reverse_pack_fn

app = App()

app.register(reverse_pack_fn)

app.run()
