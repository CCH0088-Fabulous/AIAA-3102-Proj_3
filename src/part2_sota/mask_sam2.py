class SAM2MaskGenerator:
    """Canonical Part 2 mask generator placeholder."""

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir

    def generate(self, frames):
        raise NotImplementedError(
            "SAM 2 mask generation will be implemented in Phase 4. "
            "The file is added in Phase 0 to lock the module name and interface."
        )