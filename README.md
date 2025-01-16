## conda-whl-support

## Examples

Create a conda environment with conda-whl-support:

```
conda create --prefix ./demo_env -c jjhelmus/label/conda_whl_support conda-whl-support conda=24.11.3
```

Use this to install a wheel from a static demo repo:

```
conda activate ./demo_env
./demo_env/bin/conda create --prefix test_env --channel https://jjhelmus.github.io/sample-whl-repo/repo imagesize
```

This environment listing includes `imagesize` installed from the test channel:

```
❯ ./demo_env/bin/conda list --prefix test_env
# packages in environment at /Users/jhelmus/workspace/conda-whl-support/plugin/test_env:
#
# Name                    Version                   Build  Channel
bzip2                     1.0.8                h80987f9_6
ca-certificates           2024.11.26           hca03da5_0
expat                     2.6.4                h313beb8_0
imagesize                 1.4.1                         0    https://jjhelmus.github.io/sample-whl-repo/repo
libcxx                    14.0.6               h848a8c0_0
libffi                    3.4.4                hca03da5_1
libmpdec                  4.0.0                h80987f9_0
ncurses                   6.4                  h313beb8_0
openssl                   3.0.15               h80987f9_0
pip                       24.2            py313hca03da5_0
python                    3.13.1          h4862095_100_cp313
python_abi                3.13                    0_cp313
readline                  8.2                  h1a28f6b_0
setuptools                75.1.0          py313hca03da5_0
sqlite                    3.45.3               h80987f9_0
tk                        8.6.14               h6ba3021_0
tzdata                    2024b                h04d1e81_0
wheel                     0.44.0          py313hca03da5_0
xz                        5.4.6                h80987f9_1
zlib                      1.2.13               h18a0788_1
```

Setup environments for generating and server a local repo:

```
conda create --prefix gen_env python=3.12 pip --yes
./gen_env/bin/python -m pip install pypi-simple packaging requests python-dotenv

```

Create repodata for flask and dependencies on macos-arm64:

```
cd tools
conda activate ../gen_env
python gen_repo.py
cd ..
```

Upload the repo files
Add a .env file with

```
WHEEL_SERVER_PASSWORD=PASSWORD_GOES_HERE
WHEEL_SERVER_HOST="SERVER_URL"
```

Where the server URL points to a server running https://github.com/intentionally-left-nil/conda_wheel_server

Then, execute
python upload_repo.py upload --channel=demo_flask

Flask can now be installed from this repo:

```
./demo_env/bin/conda create --prefix test_env flask=3.1.0 python=3.13 --channel http://127.0.0.1:8000/repo
```
