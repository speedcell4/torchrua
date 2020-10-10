# TorchRua

![Unit Tests](https://github.com/speedcell4/TorchRua/workflows/Unit%20Tests/badge.svg)

*Rua* derives from the Szechwanese character <ruby>挼<rt>ruá</rt></ruby> which means "pack, rumple, screw up, manipulate". TorchRua provides tons of easy-to-use functions to help you rua variable-length Tensors with `PackedSequence`!

## requirements

- Python3.7 or higher

## performance

* `reverse_packed_sequence`: O(1) forward and O(n) backward.

![reverse_pack](assets/reverse_pack.jpg)