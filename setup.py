from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='AdaptoLogit',
    packages=find_packages(),
    version='0.0.1',
    author='María Fernandez Morín; Álvaro Méndez Civieta; Rosa E. Lillo Rodriguez',
    author_email='100483781@alumnos.uc3m.es',
    license='GNU General Public License',
    zip_safe=True,
    url='https://github.com/fernandezmaria/AdaptoLogit',
    description='A package implementing adaptive lasso for logistic regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    download_url='https://github.com/fernandezmaria/AdaptoLogit/archive/refs/tags/0.0.1.tar.gz',
    keywords=['adaptive-lasso', 'logistic-regression', 'weights'],
    python_requires='>=3.5',
    install_requires=["numpy >= 1.2",
                      "scipy >= 1.7.0",
                      "scikit-learn >= 1.0"]
)
