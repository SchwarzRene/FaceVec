from setuptools import find_packages, setup

setup(
    name='facevec',
    packages=find_packages(),
    package_data = { 'facevecModels': ['facevec/utils/Models/*']},
    version='2.3.1',
    description='library to detect and vectorise faces',
    author='Schwarz Rene',
    license='MIT',
    install_requires=[ 'tensorflow', 'numpy', 'opencv-python', 'onnxruntime', 'tqdm' ],
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)