class ProPainterInpainter:
    """Canonical Part 2 inpainting placeholder."""

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir

    def inpaint(self, frames, masks):
        raise NotImplementedError(
            "ProPainter integration will be implemented in Phase 4. "
            "The file is added in Phase 0 to lock the module name and interface."
        )