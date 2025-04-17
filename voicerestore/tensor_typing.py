from torch import Tensor
from jaxtyping import Float as JFloat, Int as JInt, Bool as JBool

# Re-export the types directly
Float = JFloat[Tensor, "..."]
Int = JInt[Tensor, "..."]
Bool = JBool[Tensor, "..."]

__all__ = ["Float", "Int", "Bool"]
