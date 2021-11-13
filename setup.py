import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DSTRF",
    version="0.2",
    author="Menoua Keshishian",
    author_email="mk4011@columbia.edu",
    description="Dynamic Spectrotemporal Receptive Field (dSTRF) Analysis Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naplab/DSTRF",
    project_urls = {
        "Bug Tracker": "https://github.com/naplab/DSTRF/issues"
    },
    license='MIT',
    packages=['dynamic_strf'],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchaudio',
        'pytorch_lightning',
        'radam'
    ],
)
