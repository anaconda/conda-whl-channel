
import json



def read_repo():
    names: set[str] = set()
    name_versions = set()
    depends_names = set()

    subdirs = ["noarch", "osx-arm64"]
    for subdir in subdirs:
        repodata_filename = f"./repo/{subdir}/repodata.json"
        with open(repodata_filename) as fh:
            repodata = json.load(fh)

        for pkg, info in repodata["packages"].items():
            name = info["name"]
            version = info["version"]
            names.add(info["name"])
            name_versions.add((name, version))
            for dep in info.get("depends", []):
                dep_name = dep.split()[0]
                depends_names.add(dep_name)

    return names, name_versions, depends_names


def extra_name_to_good(name):
    name = name[3:]  # remove leading off _c_
    dep_name, extra = name.split("-with-")
    return f"{dep_name}[{extra}]"

    breakpoint()

names, name_versions, depends_names = read_repo()
selectors = set(name for name in names if name.startswith("_c_") and "-with-" not in name)
extras = set(name for name in names if name.startswith("_c_") and "-with-" in name)
wheels = names - selectors - extras

depends_not_seen = depends_names - names
extras_not_seen = set(name for name in depends_not_seen if name.startswith("_c_") and "-with-" in name)

print("names:", len(names))
print("selector names:", len(selectors))
print("extra metapackage names:", len(extras))
print("wheel package names", len(wheels))

print("depends names", len(depends_names))
print("---------------------------")
for name in sorted(extras_not_seen):
    print(extra_name_to_good(name))

