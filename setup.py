from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f.readlines()]

setup(
    name='dialektik',
    url='https://github.com/JosefAlbers/Dialektik',
    py_modules=['dialektik'],
    packages=find_packages(),
    version='0.0.1-alpha',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="Merge. Synthesize. Create. Dialektik generates new content by fusing ideas from diverse sources, revealing unexpected insights and perspectives.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Josef Albers",
    license="MIT",
    python_requires=">=3.12.3",
    install_requires=requirements,
    extra_require={
        'setup':['phi-3-vision-mlx'],
    },
    entry_points={
        "console_scripts": [
            "dialektik = dialektik:main",
        ],
    },
)
