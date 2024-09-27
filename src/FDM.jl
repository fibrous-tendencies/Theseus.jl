#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(q, Cn, Cf, Pn, Nf, sp_init)

    Q = sparse(sp_init, sp_init, q)

    # Perform computation
    result = (Cn' * Q * Cn) \ (Pn - Cn' * Q * Cf * Nf)
    return result
end


