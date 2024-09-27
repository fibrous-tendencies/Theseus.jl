#######
# contains functions for analyzing an FDM network
#######

```
Explicit solver function
```
function solve_explicit(q, Cn, Cf, Pn, Nf, sp_init)
    println("Cn size: ", size(Cn), ", type: ", typeof(Cn))
    println("Cf size: ", size(Cf), ", type: ", typeof(Cf))
    println("Pn size: ", size(Pn), ", type: ", typeof(Pn))
    println("Nf size: ", size(Nf), ", type: ", typeof(Nf))
    println("q size: ", size(q), ", type: ", typeof(q))

    Q = sparse(sp_init, sp_init, q)
    println("Q size: ", size(Q), ", type: ", typeof(Q))

    # Perform computation
    result = (Cn' * Q * Cn) \ (Pn - Cn' * Q * Cf * Nf)
    return result
end


