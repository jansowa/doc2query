import importlib
import pkgutil

import doc2query


def test_all_package_modules_import_without_model_download() -> None:
    modules = [item.name for item in pkgutil.walk_packages(doc2query.__path__, "doc2query.")]
    assert modules
    for module in modules:
        importlib.import_module(module)


def test_source_does_not_trigger_huggingface_download() -> None:
    for module in pkgutil.walk_packages(doc2query.__path__, "doc2query."):
        imported = importlib.import_module(module.name)
        source_file = getattr(imported, "__file__", None)
        if source_file and source_file.endswith(".py"):
            with open(source_file, encoding="utf-8") as handle:
                assert ".from_pretrained(" not in handle.read()
