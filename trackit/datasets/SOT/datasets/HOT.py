from trackit.datasets.common.seed import BaseSeed


class HOT_Seed(BaseSeed):
    """Seed for the HOT single object tracking dataset."""

    def __init__(self, root_path: str = None, split: str = 'train'):
        if root_path is None:
            root_path = self.get_path_from_config('HOT_PATH')
        assert split in ('train', 'test'), f'Unknown split {split}'
        self.split = split
        super().__init__(f'HOT_{split}', root_path)

    def construct(self, constructor):
        from .Impl.HOT import construct_HOT
        construct_HOT(constructor, self)
