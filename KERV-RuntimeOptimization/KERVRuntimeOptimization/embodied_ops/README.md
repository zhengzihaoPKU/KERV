# KERV Runtime Operators

The package registers 18 KERV-specific interfaces in
`torch.ops.flagos_embodied`. Fourteen mainline interfaces are available to the
BF16 release profile; four experimental interfaces remain opt-in. The source
of truth for names, categories and selection policy is
`operator_manifest.json`.

Every interface has an exact ATen fallback. `backend=auto` selects a Triton or
fused path only for supported layouts and dtypes; unsupported inputs preserve
the reference implementation.
