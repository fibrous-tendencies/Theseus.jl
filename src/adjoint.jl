function isderiving()
    return false
end

function ChainRulesCore.rrule(::typeof(isderiving))
    # Override the output during differentiation
    y = true

    function pullback(Δy)
        # No gradient w.r.t inputs (since there are none)
        return NO_FIELDS
    end

    return y, pullback
end
