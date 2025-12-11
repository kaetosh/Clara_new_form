# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:01:56 2025

@author: a.karabedyan
"""

class MissingColumnsError(Exception):
    """Вызывается, когда в DataFrame отсутствуют обязательные столбцы."""
    pass
class RowCountError(Exception):
    """Вызывается, когда в DataFrame не достаточное количество записей."""
    pass
class LoadFileError(Exception):
    """Вызывается в прочих случаях ошибок загрузки файла."""
    pass
class CancelingFileSelectionError(Exception):
    """Вызывается в случае отмены выбора файла для обучения."""
    pass
class ClassRepresentationError(Exception):
    """Ошибка, связанная с недостаточной представленностью классов."""
    pass
class ClassSampleSizeError(Exception):
    """Ошибка, возникающая при слишком малом количестве примеров в одном или нескольких классах."""
    pass
class LoadModelError(Exception):
    """Ошибка, возникающая при ошибках загрузки joblib."""
    pass
class NoFilesToDeleteError(Exception):
    """Кастомное исключение для случая, когда не найдены файлы для удаления."""
    pass

