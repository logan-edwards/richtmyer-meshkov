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
        else:
            self.N = np.size(x_init)

        self.z = x_init + 1j * y_init
        self.dzdt = dxdt_init + 1j * dydt_init
        self.circulation = circulation_init