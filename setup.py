import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='WiMANS',
    version='1.1',
    description='A Benchmark Dataset for WiFi-based Multi-user Activity Sensing',
    packages=setuptools.find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[],
    python_requires='>=3',
    zip_safe=False,
    license="MIT",
    url='https://github.com/skgopalakrishnan/WiMANS',
    include_package_data=True,
)