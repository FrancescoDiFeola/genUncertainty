from setuptools import setup, find_packages

setup(
    name="latent-uq-model-zoo",
    version="0.2.0",
    packages=find_packages(include=["latent_uq", "latent_uq.*", "src", "src.*", "inferers", "inferers.*"]),
)


if __name__ == "__main__":
    import numpy as np

    data = np.load("/Users/francescodifeola/Desktop/mr_089.npy")

