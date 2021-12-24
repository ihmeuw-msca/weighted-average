from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

about = {}
with open(path.join(here, 'weave', '__about__.py')) as about_file:
    exec(about_file.read(), about)

with open(path.join(here, 'README.md')) as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'CHANGELOG.md')) as changelog_file:
    changelog = changelog_file.read()

long_description = readme + '\n\n' + changelog

install_requires = [
    'numba',
    'numpy',
    'pandas',
]

tests_require = [
    'pytest',
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__summary__'],
    long_description=long_description,
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__url__'],
    classifiers=about['__classifiers__'],
    license=about['__license__'],
    install_requires=install_requires,
    tests_require=tests_require,
    python_requres='>=3',
    zip_safe=False,
    packages=['weave'],
    package_dir={
        'weave': 'weave',
    },
    include_package_data=True,
)
