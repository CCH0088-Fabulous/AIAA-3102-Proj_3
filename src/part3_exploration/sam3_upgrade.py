class SAM3UpgradeRefiner:
    """Canonical Part 3 refinement placeholder."""

    def __init__(self, weights_dir):
        self.weights_dir = weights_dir

    def refine(self, frames, coarse_masks):
        raise NotImplementedError(
            "SAM 3 refinement will be implemented in Phase 5. "
            "The file is added in Phase 0 to lock the module name and interface."
        )