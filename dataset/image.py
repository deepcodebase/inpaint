import os

from torchvision import datasets


class ImageDataset(datasets.DatasetFolder):

    def __init__(self, data_dir: str = '/path/to/data', transform=None):

        self.data_dir = data_dir
        self.transform = transform

        self.samples = []
        self.loader = datasets.folder.default_loader

        for r, _, fnames in sorted(
                os.walk(os.path.expanduser(self.data_dir), followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(r, fname)
                if datasets.folder.is_image_file(path):
                    self.samples.append(path)

    def __getitem__(self, index):

        path = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
        except:
            sample = None
        return sample
