from setuptools import setup

setup(
    name='keras-losses',
    version='0.1',
    packages=['keras_losses'],
    url='https://github.com/CyberZHG/keras-losses',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Some loss functions in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'Keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
