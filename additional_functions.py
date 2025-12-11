from pathlib import Path
import json
import pandas as pd
import sys
import os
import numpy as np
from typing import List
import subprocess
import platform
import ctypes
from ctypes import wintypes

from custom_errors import MissingColumnsError, RowCountError, LoadFileError, CancelingFileSelectionError, NoFilesToDeleteError


# Явно определим недостающий тип PVOID
if not hasattr(wintypes, 'PVOID'):
    wintypes.PVOID = ctypes.c_void_p

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        print(f"Running in PyInstaller temp dir: {base_path}")  # Debug
    except AttributeError:
        base_path = os.path.abspath(".")
        print(f"Running in dev mode, base path: {base_path}")  # Debug

    full_path = os.path.join(base_path, relative_path)
    normalized_path = os.path.normpath(full_path)
    print(f"Resource final path: {normalized_path}")  # Debug
    return normalized_path

def load_font_temp(font_path: str) -> bool:
    """Упрощенная версия загрузки шрифта"""
    try:
        font_path_abs = os.path.abspath(font_path)
        if not os.path.exists(font_path_abs):
            return False

        # Простая версия без Ex-функции
        result = ctypes.windll.gdi32.AddFontResourceW(font_path_abs)
        return result > 0
    except:
        return False

def unload_font_temp(font_path: str) -> bool:
    """Выгружает временный шрифт"""
    try:
        FR_PRIVATE = 0x10
        font_path_unicode = os.path.abspath(font_path)

        RemoveFontResourceEx = ctypes.windll.gdi32.RemoveFontResourceExW
        RemoveFontResourceEx.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.PVOID]
        RemoveFontResourceEx.restype = wintypes.BOOL

        if not os.path.exists(font_path_unicode):
            print(f"Файл шрифта не найден: {font_path_unicode}")
            return False

        res = RemoveFontResourceEx(font_path_unicode, FR_PRIVATE, None)
        if not res:
            raise ctypes.WinError()

        HWND_BROADCAST = 0xFFFF
        WM_FONTCHANGE = 0x001D
        ctypes.windll.user32.SendMessageW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0)
        return True

    except Exception as e:
        print(f"Ошибка выгрузки шрифта: {e}")
        return False


def check_font() -> bool:
        """Упрощенная проверка шрифта без зависимостей"""
        try:
            from matplotlib import font_manager
            return "Cascadia Code" in {f.name for f in font_manager.fontManager.ttflist}
        except:
            # Если проверка невозможна, предполагаем что шрифта нет
            return False

def check_claras_folder() -> Path:
    documents_path = Path.cwd() / 'Claras_folder'
    if not documents_path.exists():
        documents_path.mkdir(parents=True, exist_ok=True)
    return documents_path

def get_short_path(path: str | Path = None, expand_user: bool = True) -> str:
    """Возвращает сокращённый путь (относительный или абсолютный).
    
    Args:
        path: Путь в виде строки или объекта Path.
        expand_user: Заменять ли `~` на домашнюю директорию (по умолчанию True).
    
    Returns:
        Строка с удобочитаемым путём.
    """
    if not path:
        path = check_claras_folder()
    else:    
        path = Path(path).expanduser() if expand_user else Path(path)
        path = path.resolve()
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Возвращаем путь в виде строки (можно добавить замену домашней директории на ~)
        home = Path.home()
        try:
            return "~/" + str(path.relative_to(home))
        except ValueError:
            return str(path)

def open_folder_in_explorer(folder_path: Path):
    folder_path = folder_path.resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Путь {folder_path} не является директорией")

    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", str(folder_path)])
    elif system == "Darwin":  # macOS
        subprocess.run(["open", str(folder_path)])
    else:  # Linux и другие
        subprocess.run(["xdg-open", str(folder_path)])


def confusion_matrix_to_markdown(cm: np.ndarray, classes: np.ndarray) -> str:
    """
    Возвращает confusion matrix в markdown, показывая топ-10 классов по количеству истинных примеров,
    остальные объединены в класс 'Прочие'.

    Если классов 10 или меньше, выводит полную матрицу без 'Прочие'.

    Args:
        cm: квадратная матрица ошибок (numpy.ndarray)
        classes: массив имён классов (numpy.ndarray или list)

    Returns:
        markdown-строка с таблицей
    """
    n_classes = len(classes)
    support = cm.sum(axis=1)

    if n_classes <= 10:
        # Просто выводим полную матрицу без "Прочие"
        classes_new = classes
        cm_new = cm
    else:
        # Индексы топ-10 классов по поддержке
        top10_idx = np.argsort(support)[::-1][:10]
        other_idx = np.setdiff1d(np.arange(n_classes), top10_idx)

        frequent_classes = classes[top10_idx]

        size_new = 11  # 10 + 1 "Прочие"
        cm_new = np.zeros((size_new, size_new), dtype=cm.dtype)

        # Заполняем подматрицу топ-10
        for i_new, i_old in enumerate(top10_idx):
            for j_new, j_old in enumerate(top10_idx):
                cm_new[i_new, j_new] = cm[i_old, j_old]

        # Заполняем строки для "Прочие" (истинные примеры вне топ-10)
        for j_new, j_old in enumerate(top10_idx):
            cm_new[-1, j_new] = cm[other_idx, j_old].sum()

        # Заполняем столбцы для "Прочие" (прогнозы вне топ-10)
        for i_new, i_old in enumerate(top10_idx):
            cm_new[i_new, -1] = cm[i_old, other_idx].sum()

        # Элемент "Прочие" на пересечении редких классов
        cm_new[-1, -1] = cm[np.ix_(other_idx, other_idx)].sum()

        classes_new = np.append(frequent_classes, "Прочие")

    # Формируем markdown
    header = "| Истина \\ Прогноз | " + " | ".join(classes_new) + " |\n"
    separator = "|---" * (len(classes_new) + 1) + "|\n"
    rows = ""
    for i, class_name in enumerate(classes_new):
        row = f"| {class_name} | " + " | ".join(str(x) for x in cm_new[i]) + " |\n"
        rows += row
    return header + separator + rows




def find_cls_model_files() -> list[Path]:
    """
    Ищет файлы .joblib, начинающиеся на 'cls_model' в текущей директории.

    Возвращает:
        list[Path]: Список путей к найденным файлам (объекты Path).
    """
    current_dir = Path.cwd()
    return list(current_dir.glob("cls_model*.joblib"))


def load_and_validate_excel(
    file_path: Path,
    required_columns: set,
    min_rows: int = None
) -> pd.DataFrame:
    """
    Загружает Excel-файл и проверяет наличие заданных столбцов и (опционально) минимальное количество строк.

    Параметры:
        file_path (Path): Путь к файлу Excel,
        required_columns (set): Множество обязательных столбцов. По умолчанию {'Наименование', 'Группа'},
        min_rows (int или None): Минимальное количество строк. Если None — проверка не выполняется.

    Возвращает:
        DataFrame: Если проверки пройдены.

    Исключения:
        CancelingFileSelectionError: если file_path не задан.
        MissingColumnsError: если отсутствуют обязательные столбцы.
        RowCountError: если количество строк меньше min_rows.
        LoadFileError: при ошибках чтения файла или других исключениях.
    """
    try:
        if not file_path:
            raise CancelingFileSelectionError("Отмена выбора файла")

        df = pd.read_excel(file_path)

        # Проверка столбцов
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise MissingColumnsError(f"Ошибка: отсутствуют столбцы {missing}")

        # Проверка количества строк (если задана)
        if min_rows is not None and len(df) < min_rows:
            raise RowCountError(f"Ошибка: в файле только {len(df)} строк (требуется >= {min_rows})")

        if df.empty:
            raise RowCountError('Таблица пустая')

        return df
    except FileNotFoundError:
        raise LoadFileError(f"Ошибка загрузки файла: {file_path.name} не найден")

    except Exception as e:
        raise LoadFileError(f"Ошибка загрузки файла: {e}")

def open_excel_file(path: Path):
    """
    Открывает Excel-файл в приложении Excel по заданному пути.

    Args:
        path (Path): Путь к файлу Excel.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Приводим путь к строке с абсолютным путем
    file_path = str(path.resolve())
    try:
        if sys.platform == "win32":
            # Windows: используем startfile
            os.startfile(file_path)
    except Exception:
        raise LoadFileError("Не удалось открыть файл.")
        
def delete_files_by_type(file_types: List[int]) -> None:
    """
    Удаляет файлы в папке Claras_folder по указанным типам.
    Вызывает NoFilesToDeleteError, если файлы не найдены.

    Параметры:
        file_types: Список чисел, где:
            [0] - удалить все .xlsx файлы
            [1] - удалить все .joblib файлы
            [0, 1], [1, 0]- удалить оба типа файлов

    Исключения:
        ValueError: Если передан недопустимый тип файла
        NoFilesToDeleteError: Если не найдены файлы для удаления
        PermissionError: Если нет прав на удаление файлов
        OSError: При других ошибках файловой системы
    """
    valid_inputs = [[0], [1], [0, 1], [1, 0]]
    if file_types not in valid_inputs:
        raise ValueError(f"Недопустимый список типов. Допустимые значения: {valid_inputs}")

    extensions = []
    if 0 in file_types:
        extensions.append('.xlsx')
    if 1 in file_types:
        extensions.append('.joblib')

    folder_path = check_claras_folder()

    files_found = False

    try:
        for filename in os.listdir(folder_path):
            if any(filename.endswith(ext) for ext in extensions):
                files_found = True
                file_path = folder_path / filename
                try:
                    os.remove(file_path)
                    print(f"Удалён файл: {file_path}")
                except PermissionError as e:
                    raise PermissionError(f"Нет прав на удаление файла {file_path}") from e
                except OSError as e:
                    raise OSError(f"Ошибка при удалении файла {file_path}") from e

        if not files_found:
            raise NoFilesToDeleteError("Не найдено файлов для удаления с указанными расширениями")

    except OSError as e:
        raise OSError(f"Ошибка при чтении содержимого папки {folder_path}") from e




SETTINGS_FILE = 'settings.json'

def save_state(value: bool):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump({'toggle_state': value}, f)

def load_state() -> bool:
    if not os.path.exists(SETTINGS_FILE):
        return False  # Значение по умолчанию
    
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f).get('toggle_state', False)
