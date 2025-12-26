# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:37:18 2025

@author: a.karabedyan
"""
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

from textual import work, on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Container, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, LoadingIndicator, Markdown, SelectionList, Static, Switch, Footer

from additional_functions import load_and_validate_excel, open_excel_file, delete_files_by_type, check_claras_folder, get_short_path, save_state, load_state
from widgets import ExcelDirectoryTree, JoblibDirectoryTree
from configuration import MIN_SAMPLES
from complementNB import AssetClassifier

from configuration import REQUIRED_COLUMNS
from data_text import TEXT_HELP


class LoaderIndicatorCustom(ModalScreen):
    def compose(self) -> ComposeResult:
        yield Grid(Label("Идет обработка данных."),
                   LoadingIndicator(),
                   id='grid-loader_indicator',
                   )

class HelpScreen(ModalScreen):
    """Экран со справкой"""
    
    BINDINGS = [("escape", "dismiss", "Закрыть")] # Используем dismiss вместо action_close
    
    def compose(self) -> ComposeResult:
        yield Markdown(TEXT_HELP)
        yield Footer()
    
    def on_mount(self) -> None:
        self.border_title = "Справка - нажмите Escape для закрытия"

class TrainingWarningModal(ModalScreen):
    """
    Окно с предупреждением о том, что нужен файл с данными
    для обучения определенного формата. Кнопки - Продолжить и Отмена.
    При нажатии Продолжить - окно FileSelectTrainModal с выбором файла для
    обучения модели.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:
        yield Grid(Label(f"""Убедитесь, что excel файл с обучающей выборкой:
✅ расположен в папке {get_short_path()} (ctrl+o откроет папку) 
✅ соотвествует требованиям раздела -Обязательные условия- """,
                         id="label-training-warning-modal"),
                   Button("Продолжить", variant="success", id="button-continue-training-warning-modal"),
                   Button("Отмена", variant="error", id="button-cancel-training-warning-modal"),
                   id='grid-training-warning-modal'
                         )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-continue-training-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
            self.app.push_screen(FileSelectTrainModal())
        elif event.button.id == "button-cancel-training-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class FileSelectTrainModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с данными
    для обучения. Кнопки Выбрать и Отмена. После выбора файла - запуск обучения
    модели и последующий вывод отчета по обучению (окно PrintReportModal)
    """
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:
        yield Grid(Label("Выберите файл Excel для обучения (.xlsx):",
                         id="label-file-select-train-modal"),
                   ExcelDirectoryTree(check_claras_folder(), id="tree-file-select-train-modal"),
                   Button("Выбрать", variant="success", id="button-select-tree-file-select-train-modal", disabled=True),
                   Button("Отмена", variant="error", id="button-cancel-tree-file-select-train-modal"),
                   id='grid-file-select-train-modal'
                   )


    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-tree-file-select-train-modal").disabled = True

    def on_directory_tree_file_selected(self, event: ExcelDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".xlsx":
            self.app.selected_path_file_train = event.path
            self.query_one("#button-select-tree-file-select-train-modal").disabled = False
            self.query_one("#label-file-select-train-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл Excel (.xlsx) с данными для обучения")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-tree-file-select-train-modal":
            self.app.notify("Ожидайте завершение обучения. Процесс может занять несколько минут.", title="Статус")
            self.app.push_screen(LoaderIndicatorCustom())
            self.process_file(self.app.selected_path_file_train)  # Запускаем фоновую задачу   
        elif event.button.id == "button-cancel-tree-file-select-train-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, file) -> None:
        try:
            df = load_and_validate_excel(file, required_columns=set(REQUIRED_COLUMNS), min_rows=MIN_SAMPLES)
            # df = df.sample(n=1000, random_state=42) # для тестирования
            # Инициализация и обучение
            self.app.classifier = AssetClassifier()
            self.app.report = self.app.classifier.train(df, text_column=REQUIRED_COLUMNS[0], target_column=REQUIRED_COLUMNS[1])
            self.app.call_from_thread(self.on_success)  # Возвращаемся в основной поток

        except Exception as e:
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify("Обучение завершено, сохраните модель, если результаты удовлетворительные.", title="Статус")
        self.app.push_screen(PrintReportModal(self.app.report))

    def on_error(self, error: str) -> None:
        self.app.pop_screen()  # Закрываем индикатор
        self.app.notify(error, title="Ошибка", severity='error', timeout=15)

class PrintReportModal(ModalScreen):
    """
    Выводит на экран результат обучения модели (метрики, описание, таблица).
    Кнопки Сохранить модель (выводит окно SetNameModelModal) и Отмена.
    """
    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        yield Grid(
            Label('📋 Отчет по результатам обучения 📋', id="label-print-report-modal"),
            ScrollableContainer(
                Markdown(self.message, id='markdown-print-report-modal'),
                id='scroll-container-print-report-modal'
            ),
            Grid(Button("Сохранить модель", variant="success", id="button-save-print-report-modal"),
                 Button("Отмена", variant="error", id="button-cancel-print-report-modal"),
                 classes='grid-buttons'),
            id='grid-print-report-modal'
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-save-print-report-modal":
            self.app.notify("Дайте модели осмысленное название, чтобы позже легко её идентифицировать. Например: Классификатор запасов (розница 2024) или ОС + НМА для IT-компаний",
                            title="Важно", severity='warning', timeout=10)
            self.app.push_screen(SetNameModelModal())
        elif event.button.id == "button-cancel-print-report-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class SetNameModelModal(ModalScreen):
    """
    Окно для ввода имени и сохранения модели. Кнопки Сохранить и Отмена.
    Нажатие любой кнопки закрывает текущие окна.
    """
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("""Модель успешно обучена. Введите имя модели и нажмите -Сохранить-""",
                  id="label-set-name-model-modal"),
            Input(placeholder="Имя модели", type="text", id="input-set-name-model-modal"),
            Button("Сохранить", variant="success", id="button-save-set-name-model-modal"),
            Button("Отмена", variant="error", id="button-cancel-set-name-model-modal"),
            id="grid-set-name-model-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-save-set-name-model-modal":
            # Сохранение модели
            name_model = self.query_one('#input-set-name-model-modal').value
            self.app.classifier.save_model(name_model)
            self.app.notify("Модель сохранена. Можно классифицировать.", title="Статус")
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
        elif event.button.id == "button-cancel-set-name-model-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()


class PredictWarningModal(ModalScreen):
    """
    Окно с предупреждением о том, что нужен файл с данными
    для прогноза определенного формата. Кнопки - Продолжить и Отмена.
    При нажатии Продолжить - окно FileSelectPredictModal с выбором файла для
    прогноза моделью.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:
        yield Grid(
            Label(f"""Убедитесь, что excel файл с данными для классификации:
✅ расположен в папке {get_short_path()} (ctrl+o откроет папку)  
✅ соотвествует требованиям раздела -Классификация-         

Убедитесь, что предварительно обученная модель:
✅ расположена в папке {get_short_path()} (ctrl+o откроет папку).""",
                  id="label-predict-warning-modal"),
            Button("Продолжить", variant="success", id="button-continue-predict-warning-modal"),
            Button("Отмена", variant="error", id="button-cancel-predict-warning-modal"),
            id="grid-predict-warning-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-continue-predict-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
            self.app.push_screen(FileSelectPredictModal())
        elif event.button.id == "button-cancel-predict-warning-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

class FileSelectPredictModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с данными
    для прогноза. Кнопки Выбрать и Отмена. После выбора файла - окно FileSelectModelModal
    для выбора обученной модели для классификации.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:

        yield Grid(Label("Выберите файл Excel для классификации (.xlsx):",
                         id='label-file-select-predict-modal'),
                   ExcelDirectoryTree(check_claras_folder(), id="tree-file-select-predict-modal"),
                   Button("Выбрать", variant="success", id="button-select-file-select-predict-modal", disabled=True),
                   Button("Отмена", variant="error",id="button-cancel-file-select-predict-modal"),
                   id='grid-file-select-predict-modal'
                   )

    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-file-select-predict-modal").disabled = True
        # self.query_one("#tree-file-select-predict-modal").ICON_FILE = '◼ '
        # self.query_one("#tree-file-select-predict-modal").ICON_NODE = '▼ '
        # self.query_one("#tree-file-select-predict-modal").ICON_NODE_EXPANDED = '▶ '

    def on_directory_tree_file_selected(self, event: ExcelDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".xlsx":
            self.app.selected_path_file_predict = event.path
            self.query_one("#button-select-file-select-predict-modal").disabled = False
            self.query_one("#label-file-select-predict-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл Excel (.xlsx) с данными для классификации")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-file-select-predict-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
            self.app.push_screen(FileSelectModelModal())
        elif event.button.id == "button-cancel-file-select-predict-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

class FileSelectModelModal(ModalScreen[Optional[Path]]):
    """
    Окно с деревом файлов текущей папки для выбора файла с обученной моделью
    для прогноза. Кнопки Выбрать и Отмена. После выбора файла - запуск процесса
    классификации, после чего вызов окна PrintPredictModal с результатами классификации.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:
        yield Grid(Label("Выберите файл модели для классификации (.joblib):",
                         id='label-file-select-model-modal'),
                   JoblibDirectoryTree(check_claras_folder(), id="tree-file-select-model-modal"),
                   Button("Выбрать", variant="success", id="button-select-file-select-model-modal", disabled=True),
                   Button("Отмена", variant="error", id="button-cancel-file-select-model-modal"),
                   id='grid-file-select-model-modal'
                   )

    def on_mount(self):
        # Отключаем кнопку пока файл не выбран
        self.query_one("#button-select-file-select-model-modal").disabled = True
        self.query_one("#tree-file-select-model-modal").ICON_FILE = '🎓 '
        # self.query_one("#tree-file-select-model-modal").ICON_NODE = '▼ '
        # self.query_one("#tree-file-select-model-modal").ICON_NODE_EXPANDED = '▶ '

    def on_directory_tree_file_selected(self, event: JoblibDirectoryTree.FileSelected):
        """Обработчик выбора файла"""
        if event.path.suffix == ".joblib":
            self.app.selected_path_file_joblib = event.path
            self.query_one("#button-select-file-select-model-modal").disabled = False
            self.query_one("#label-file-select-model-modal").update(f"Выбран: {event.path.name}")
        else:
            self.notify("Выберите файл модели (.joblib)")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-select-file-select-model-modal":
            self.app.push_screen(LoaderIndicatorCustom())
            self.process_file(file_predict_data=self.app.selected_path_file_predict,
                              model_path= self.app.selected_path_file_joblib)  # Запускаем фоновую задачу
        elif event.button.id == "button-cancel-file-select-model-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, file_predict_data, model_path) -> None:
        try:
            df = load_and_validate_excel(file_predict_data, required_columns={REQUIRED_COLUMNS[0]})
            loaded_classifier = AssetClassifier.load_model(model_path)
            # Предсказание на новых данных
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            claras_folder = check_claras_folder()
            self.app.result = claras_folder / f'Результат_класс_по_{model_path.stem}_{timestamp}.xlsx'
            # self.app.result = f'Результат_класс_по_{model_path.stem}_{timestamp}.xlsx'
            _, md_table = loaded_classifier.predict(df, return_proba=True, output_file=self.app.result)
            self.app.call_from_thread(self.on_success, df, md_table)  # Возвращаемся в основной поток
        except Exception as e:
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self, df: pd.DataFrame, md_table) -> None:

        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify(f"Файл {self.app.result.name} успешно выгружен!", title="Статус")
        self.app.push_screen(PrintPredictModal(md_table, Path(self.app.result)))

    def on_error(self, error: str) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify(error, title="Ошибка", severity='error', timeout=15)


class PrintPredictModal(ModalScreen):
    """
    Окно с результатами классификации. Кнопки Открыть... и Отмена.
    Открыть запускает сохраненный файл excel с результатами классификации и закрывает
    текущие окна.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def __init__(self, message: str, path_file_result_classification: Path, **kwargs):
        super().__init__(**kwargs)
        self.message = message
        self.path_file = path_file_result_classification

    def compose(self) -> ComposeResult:
        yield Grid(
            Label('📑 Отчет по результатам классификации (отображено не более 100 первых строк) 📑', id="label-print-predict-modal"),
            ScrollableContainer(
                Markdown(self.message, id='markdown-print-predict-modal'),
                id='scroll-container-print-predict-modal'
            ),
            Grid(Button("Открыть файл", variant="success", id="button-open-print-predict-modal"),
                 Button("Отмена", variant="error", id="button-cancel-print-predict-modal"),
                 classes='grid-buttons'),
            id='grid-print-predict-modal'
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "button-open-print-predict-modal":
            self.app.notify("Открываем файл с результатами классификации...", title="Статус")
            open_excel_file(self.path_file)
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()
        elif event.button.id == "button-cancel-print-predict-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

class ClearDirModal(ModalScreen[Optional[Path]]):
    """
    Окно с выбором типа файлов (.xlsx или .joblib) для удаления из текущей директории. Кнопки Очистить и Отмена.
    """
    
    BINDINGS = [("ctrl+o", "open_dir", "Открыть папку")]
    def action_open_dir(self):
        self.app.action_open_dir()
    
    def compose(self) -> ComposeResult:
        yield Grid(Label(f"Выберите файлы для удаления из {get_short_path()}:",
                         id="label-clear-dir-modal"),
                   SelectionList(("файлы Excel (.xlsx)", 0, True), ("файлы модели (.joblib)", 1),),
                   Button("Очистить", variant="success", id="button-clear-dir-modal"),
                   Button("Отмена", variant="error", id="button-cancel-clear-dir-modal"),
                   id='grid-clear-dir-modal'
                   )
    @on(SelectionList.SelectedChanged)
    def handle_select_sheet(self):
        self.app.selected_files_for_clear = self.query_one(SelectionList).selected
        if not self.app.selected_files_for_clear:
            self.query_one('#button-clear-dir-modal').disabled = True
        else:
            self.query_one('#button-clear-dir-modal').disabled = False

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-clear-dir-modal":
            self.app.notify("Ожидайте очистку папки.", title="Статус")
            self.app.push_screen(LoaderIndicatorCustom())
            self.app.selected_files_for_clear = self.query_one(SelectionList).selected
            self.process_file(self.app.selected_files_for_clear)  # Запускаем фоновую задачу
        elif event.button.id == "button-cancel-clear-dir-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

    @work(thread=True)  # Запускаем в отдельном потоке, чтобы не блокировать UI
    def process_file(self, files_for_clear) -> None:
        try:
            delete_files_by_type(files_for_clear)
            self.app.call_from_thread(self.on_success)  # Возвращаемся в основной поток
        except Exception as e:
            self.app.call_from_thread(self.on_error, str(e))  # Обработка ошибок

    def on_success(self) -> None:
        self.app.notify("Папка очищена", title="Статус")
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()

    def on_error(self, error: str) -> None:
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        self.app.notify(error, title="Ошибка", severity='error', timeout=15)

class SettingsModal(ModalScreen):
    """
    Окно с настройками:
        - показывать напоминание про установку шрифта Cascadian
    """
    
    def compose(self) -> ComposeResult:
            yield Container(
                Horizontal(
                    Static("Напоминание про установку шрифта Cascadia при запуске:     ", id='static-settings-modal'),
                    Switch(value=load_state(), id='switch-settings-modal'),
                    id="horizontal-settings-modal",
                    ),
                Button("Закрыть", variant="success", id="button-settings-modal"),
                id="container-settings-modal"
            )
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        save_state(event.switch.value)
    
    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "button-settings-modal":
            while len(self.app.screen_stack) > 1:
                self.app.pop_screen()

   