"""Full (native diff) module: fused merge add (bypass unsupported by design)."""

from ..dispatch import eager_under_dynamo
from .norms import AddScaledFn


@eager_under_dynamo
def full_diff_weight(org_weight, diff, multiplier=1.0, backend=None):
    return AddScaledFn.apply(org_weight, diff, float(multiplier), backend)
