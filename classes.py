import numpy as np

class VortexSheet:
    N: int
    z: list
    dzdt: list
    circulation: list
    sheet_strength: list
    def __init__(
        self,
        x_init,
        y_init,
        dxdt_init,
        dydt_init,
        circulation_init
        ):
        if(np.size(x_init) != np.size(y_init)):
            self.N = 0
            return(-1)
        self.N = np.size(x_init)

        self.z = x_init + 1j * y_init
        self.dzdt = dxdt_init + 1j * dydt_init
        self.circulation = circulation_init    
        return(0)

    def circulation_to_sheet_strength(
        self,
        ds
        ):
        self.sheet_strength = self.circulation / ds
        return(0)

    def sheet_strength_to_circulation(
        self,
        ds
        ):
        self.circulation = self.sheet_strength * ds
        return(0)