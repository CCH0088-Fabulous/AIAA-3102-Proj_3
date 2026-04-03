class ControlNetInpainter:
    """Canonical Part 3 diffusion placeholder."""

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir

    def inpaint(self, frames, masks):
        raise NotImplementedError(
            "Diffusion-based refinement will be implemented in Phase 5. "
            "The file is added in Phase 0 to lock the module name and interface."
        )