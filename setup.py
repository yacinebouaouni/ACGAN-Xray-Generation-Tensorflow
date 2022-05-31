import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AC-GAN-Covid",
    version="0.0.1",
    author="Yacine Bouaouni",
    author_email="yacinebouaouni1998@gmail.com",
    description="AC-GAN implementation for Covid and healthy X-rays.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yacinebouaouni/ACGAN-Xray-Generation-Tensorflow",
    project_urls={
        "Bug Tracker": "https://github.com/yacinebouaouni/ACGAN-Xray-Generation-Tensorflow/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"ACGAN-Xray-Generation-Tensorflow": "src"},
    # where="src"
    packages=setuptools.find_packages(),
    python_requires=">=3.7.13",
)

print(setuptools.find_packages())