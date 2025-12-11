# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:31:12 2025

@author: a.karabedyan
"""
from pathlib import Path
from typing import Iterable
from textual.widgets import DirectoryTree



class BaseFilteredDirectoryTree(DirectoryTree):
    """Базовый класс с общей логикой фильтрации"""

    def __init__(self, path: str | Path, **kwargs):
        super().__init__(path, **kwargs)

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            path for path in paths
            if not path.name.startswith(".")
            and not path.name.startswith("~")  # исключаем имена, начинающиеся с ~
            and (path.is_dir() or self.should_include_file(path))
        ]


    def should_include_file(self, path: Path) -> bool:
        raise NotImplementedError


class ExcelDirectoryTree(BaseFilteredDirectoryTree):
    def should_include_file(self, path: Path) -> bool:
        return path.suffix.lower() == '.xlsx'


class JoblibDirectoryTree(BaseFilteredDirectoryTree):
    def should_include_file(self, path: Path) -> bool:
        return (
            path.suffix.lower() == '.joblib' and
            '_model_' in path.name.lower()
        )
