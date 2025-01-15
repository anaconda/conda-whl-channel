import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse


app = FastAPI()


def whl_pypi_url(filename):
    # https://packaging.python.org/en/latest/specifications/binary-distribution-format/#binary-distribution-format
    # could also contain build_tag but not allowed on PyPI
    raw_name, _, python_tag = filename.split("-")[:3]
    # PEP 503
    name = re.sub(r"[-_.]+", "-", raw_name).lower()
    host = 'https://files.pythonhosted.org'
    return f'{host}/packages/{python_tag}/{name[0]}/{name}/{filename}'


@app.get("/repo/{subdir}/{filename}")
async def get_noarch_file(subdir: str, filename: str):
    if filename == "repodata.json":
        return FileResponse(f"repo/{subdir}/repodata.json")
    elif filename.startswith("_c"):
        return FileResponse(f"repo/sample-1.0-0.tar.bz2")
    elif filename.endswith(".whl"):
        try:
            url = whl_pypi_url(filename)
            return RedirectResponse(url)
        except:
            raise HTTPException(status_code=404, detail="whl cannot be redirected")
    raise HTTPException(status_code=404, detail="Item not found")
