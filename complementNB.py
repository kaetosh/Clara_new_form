from collections import Counter

import pickle
import pandas as pd
import numpy as np
import joblib
import re
from pymorphy3 import MorphAnalyzer
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from configuration import STOPWORDS_RU, REQUIRED_COLUMNS, MIN_SAMPLES
from additional_functions import confusion_matrix_to_markdown, check_claras_folder
from custom_errors import RowCountError, ClassRepresentationError, ClassSampleSizeError, LoadModelError


class TextCleaner(BaseEstimator, TransformerMixin):
    """Кастомный трансформер для очистки текста"""
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set()
        self.morph = MorphAnalyzer()
    
    def lemmatize(self, word):
        # Пропускаем короткие слова и аббревиатуры
        if len(word) <= 2 or word.isupper():
            return word
        return self.morph.parse(word)[0].normal_form
    
    def lemmatize_word(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\d+|[{}]'.format(re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»—')), ' ', text)
        text = ' '.join(text.split())
        
        # words = [self.lemmatize_word(word) for word in text.split() 
        #         if word not in self.stop_words]
        words = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(words).strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str).apply(self.clean_text)

class AssetClassifier:
    """Класс для классификации активов компании по наименованиям"""

    def __init__(self,
                 max_features=30000,
                 test_size=0.2,
                 random_state=42,
                 n_jobs=-1):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.model = None
        self.stop_words_russian = STOPWORDS_RU  # Пример стоп-слов

    def _is_sample_suitable(self,
                            X,
                            y,
                            label_encoder=None,
                            min_samples=MIN_SAMPLES,
                            min_classes=2,
                            min_samples_per_class=10):
        """
        Проверка, подходит ли выборка для обучения.

        Параметры:
        - X: признаки (тексты)
        - y: метки (закодированные числа)
        - label_encoder: экземпляр LabelEncoder для декодирования меток
        - min_samples: минимальное общее количество образцов
        - min_classes: минимальное количество уникальных классов
        - min_samples_per_class: минимальное количество примеров для каждого класса

        Возвращает:
        - True, если выборка подходит, иначе False
        """
        # Проверка общего количества образцов
        if len(X) < min_samples:
            raise RowCountError(f"(!) Выборка слишком мала (строки={len(X)} < {min_samples}).")

        # Проверка количества уникальных классов
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < min_classes:
            raise ClassRepresentationError(f"(!) Слишком мало классов (n_classes={len(unique_classes)} < {min_classes}).")

        # Проверка, что в каждом классе достаточно примеров
        if np.any(class_counts < min_samples_per_class):
            # Декодируем числовые метки в оригинальные названия (если передан label_encoder)
            if label_encoder is not None:
                problematic_classes = {
                    label_encoder.inverse_transform([cls])[0]: int(count)
                    for cls, count in zip(unique_classes, class_counts)
                    if count < min_samples_per_class
                }
            else:
                problematic_classes = {
                    int(cls): int(count)
                    for cls, count in zip(unique_classes, class_counts)
                    if count < min_samples_per_class
                }
            raise ClassSampleSizeError(f"(!) Некоторые классы содержат слишком мало примеров: {problematic_classes} < {min_samples_per_class}.")

        # Проверка, что после векторизации останутся признаки
        vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
        try:
            X_vec = vectorizer.fit_transform(X)
            if X_vec.shape[1] == 0:
                print("(!) После векторизации не осталось признаков. Попробуйте изменить `min_df`/`max_df`.")
                return False
        except ValueError as e:
            print(f"(!) Ошибка векторизации: {e}")
            return False

        return True

    def train(self, df, text_column=REQUIRED_COLUMNS[0], target_column=REQUIRED_COLUMNS[1]):
        """
        Обучение модели на предоставленных данных
        Args:
            df: DataFrame с данными
            text_column: название колонки с текстом
            target_column: название колонки с целевой переменной
        """
        try:
            # Подготовка данных
            df = df.dropna(subset=REQUIRED_COLUMNS, how='any')
            X = df[text_column].astype(str)
            y = self.label_encoder.fit_transform(df[target_column])

            if not self._is_sample_suitable(X, y, self.label_encoder):
                raise ValueError("Выборка не подходит для обучения")

            # Разделение на train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # Создание пайплайна
            pipeline = make_pipeline(
                TextCleaner(stop_words=self.stop_words_russian),
                TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    analyzer='word',
                    min_df=2,
                    max_df=0.9
                ),
                ComplementNB()
            )

            # Параметры для подбора
            param_dist = {
                'tfidfvectorizer__max_features': [15000, 20000],
                'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
                'tfidfvectorizer__min_df': [2, 3, 5],
                'complementnb__alpha': np.logspace(-5, 0, 6),
                'complementnb__norm': [True, False],
            }

            # Поиск лучших параметров
            search = RandomizedSearchCV(
                pipeline,
                param_dist,
                n_iter=20,
                cv=5,
                n_jobs=self.n_jobs,
                verbose=0,
                random_state=self.random_state,
                scoring='f1_weighted'
            )

            search.fit(X_train, y_train)
            self.model = search.best_estimator_

            # Оценка модели
            report = self.generate_training_report(X_test, y_test)

            return report

        except Exception as e:
            print(f"\nОшибка при обучении: {str(e)}")
            raise


    def save_model(self, path_prefix=None):
        """Сохранение модели и кодировщика в папку, возвращаемую check_claras_folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = path_prefix or "asset_classifier"

        folder_path = check_claras_folder()

        model_filename = f"{prefix}_model_{timestamp}.joblib"
        full_path = folder_path / model_filename

        model_components = {
            'model': self.model,
            'vectorizer': self.label_encoder
        }

        joblib.dump(model_components, full_path)

        return str(full_path)

    @classmethod
    def load_model(cls, model_path):
        """Загрузка сохраненной модели"""
        try:
            loaded = joblib.load(model_path)
        except (EOFError,
                pickle.UnpicklingError,
                ModuleNotFoundError,
                RuntimeError,
                pickle.PicklingError,
                MemoryError) as e:
            raise LoadModelError(f"Ошибка загрузки модели: файл повреждён или имеет неверный формат: {e}")

        classifier = cls()
        classifier.model = loaded['model']
        classifier.label_encoder = loaded['vectorizer']
        return classifier

    def predict(self, new_data, text_column=REQUIRED_COLUMNS[0], return_proba=False, output_file=None):
        """
        Предсказание классов для новых данных
        Args:
            new_data: DataFrame или список строк
            text_column: название колонки с текстом (если new_data - DataFrame)
            return_proba: возвращать вероятности классов
            output_file: путь для сохранения результатов (если указан)
        Returns:
            tuple:
                - DataFrame с предсказаниями и исходными данными
                - str с markdown-таблицей первых 100 строк с дополнительным текстом
        """
        import numpy as np

        try:
            # Создаем копию входных данных
            if isinstance(new_data, pd.DataFrame):
                result = new_data.copy()
                texts = result[text_column].astype(str)
            else:
                texts = pd.Series(new_data).astype(str)
                result = pd.DataFrame({text_column: texts})

            if return_proba:
                # Получаем вероятности для всех классов
                proba = self.model.predict_proba(texts)
                # Переводим вероятности в проценты и округляем
                proba_percent = (proba * 100).round(0).astype(int)
                # Добавляем знак '%' к значениям
                proba_percent_str = proba_percent.astype(str) + '%'
                proba_df = pd.DataFrame(proba_percent_str, columns=[f'вероятность_{cls}' for cls in self.label_encoder.classes_])
                result = pd.concat([result, proba_df], axis=1)

                # Предсказанные классы — максимальная вероятность
                max_idx = np.argmax(proba, axis=1)
                preds = max_idx
                result['прогнозируемая_группа'] = self.label_encoder.inverse_transform(preds)

                # Вероятность для предсказанного класса в процентах
                probs_percent = proba_percent[np.arange(len(proba)), max_idx]
                result['вероятность'] = probs_percent.astype(str) + '%'

                # Средняя вероятность верного предсказания (по всем строкам)
                avg_proba = probs_percent.mean()
                avg_proba_str = f"{avg_proba:.1f}%"

            else:
                # Получаем предсказанные классы
                preds = self.model.predict(texts)
                result['прогнозируемая_группа'] = self.label_encoder.inverse_transform(preds)
                # В отсутствие вероятностей ставим пустую строку
                result['вероятность'] = ''
                avg_proba_str = 'N/A'

            # Формируем markdown-таблицу первых 100 строк
            md_lines = []

            # Заголовок со средней вероятностью (если есть)
            md_lines.append(f"**📊 Средняя вероятность предсказания:** {avg_proba_str}\n")

            # Интерпретация результатов
            md_lines.append("**Интерпретация результатов**")
            md_lines.append("> **Вероятность >80%** - высокая достоверность\n")
            md_lines.append("> **Вероятность 60-80%** - рекомендуется проверка\n")
            md_lines.append("> **Вероятность <60%** - модель не уверена, требуется ручная классификация\n")

            # Заголовок таблицы
            header = f"| {text_column.capitalize()} | Прогнозируемая группа | Вероятность |"
            separator = f"|{'-' * (len(text_column)+2)}|----------------------|-------------|"
            md_lines.append(header)
            md_lines.append(separator)

            for _, row in result.head(100).iterrows():
                name = str(row[text_column])
                pred = str(row['прогнозируемая_группа'])
                prob = str(row['вероятность'])
                md_lines.append(f"| {name} | {pred} | {prob} |")

            markdown_table = "\n".join(md_lines)

            # Сохранение в файл (если нужно)
            # claras_folder = check_claras_folder()
            if output_file:
                # # Добавляем расширение, если его нет
                # if not output_file.endswith('.xlsx'):
                #     output_file += '.xlsx'
                
                # Создаем полный путь к файлу
                # output_path = claras_folder / output_file
                
                # Создаем папку, если она не существует
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Сохраняем файл
                result.to_excel(output_file, index=False)
                print(f"Результаты сохранены в: {output_file}")

            return result, markdown_table

        except Exception as e:
            print(f"Ошибка при предсказании: {str(e)}")
            raise




    def generate_training_report(self, X_test, y_test):
        """
        Формирует подробный отчет по итогам обучения модели с метриками, confusion matrix и classification report.
        При указании output_pdf_path сохраняет отчет и график в PDF.

        Args:
            X_test (pd.Series или список): Тестовые тексты.
            y_test (array-like): Истинные метки (числовые, закодированные LabelEncoder).

        Returns:
            str: Отчет в формате markdown.
        """


        if self.model is None:
            raise ValueError("Модель не обучена")

        # Предсказания
        y_pred = self.model.predict(X_test)

        # Декодируем метки
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # Общая точность
        accuracy = np.mean(y_pred == y_test)
        accuracy_pct = int(round(accuracy * 100))

        # Средняя вероятность верного предсказания
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_test)
            correct_class_indices = [list(self.label_encoder.classes_).index(label) for label in y_test_decoded]
            correct_probas = [proba[i, idx] for i, idx in enumerate(correct_class_indices)]
            avg_correct_proba = int(round(np.mean(correct_probas) * 100))
        else:
            avg_correct_proba = None

        # Распределение по группам (истинные метки)
        group_counts = Counter(y_test_decoded)
        total = len(y_test_decoded)

        # Качество распознавания по группам
        group_report = {}
        for group in self.label_encoder.classes_:
            total_group = group_counts.get(group, 0)
            idxs = [i for i, val in enumerate(y_test_decoded) if val == group]
            if total_group == 0:
                continue
            correct = sum(1 for i in idxs if y_pred_decoded[i] == group)
            errors = [y_pred_decoded[i] for i in idxs if y_pred_decoded[i] != group]
            error_counts = Counter(errors)
            common_errors = error_counts.most_common(3)
            group_report[group] = {
                'total': total_group,
                'correct': correct,
                'accuracy_pct': int(round(correct / total_group * 100)),
                'common_errors': common_errors
            }

        # Формируем markdown отчет
        lines = []
        lines.append("### Основные метрики")
        lines.append(f"**🎯 Общая точность:** {accuracy_pct}%")
        if avg_correct_proba is not None:
            lines.append(f"\n**📊 Средняя вероятность верного предсказания:** {avg_correct_proba}%")
        
        lines.append("\n### Распределение по группам")
        
        # Заголовок таблицы с новыми столбцами
        lines.append("| Группа | Количество | Доля | Точность | Средняя вероятность верного предсказания |")
        lines.append("|---|---|---|---|---|")
        
        total = len(y_test_decoded)
        
        for group, count in group_counts.most_common():
            pct = int(round(count / total * 100))
            accuracy_group = group_report[group]['accuracy_pct'] if group in group_report else 0
        
            # Средняя вероятность верного предсказания по группе
            if avg_correct_proba is not None and hasattr(self.model, 'predict_proba'):
                # Индексы объектов этой группы
                idxs = [i for i, val in enumerate(y_test_decoded) if val == group]
                # Индексы классов для этой группы
                class_idx = list(self.label_encoder.classes_).index(group)
                # Вероятности для правильного класса для объектов группы
                group_probas = [proba[i, class_idx] for i in idxs]
                avg_group_proba = int(round(np.mean(group_probas) * 100)) if group_probas else 0
            else:
                avg_group_proba = "N/A"
        
            lines.append(f"| {group} | {count} | {pct}% | {accuracy_group}% | {avg_group_proba}% |")


        # Confusion matrix
        classes = self.label_encoder.classes_
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=classes)

        md_table = confusion_matrix_to_markdown(cm, classes)

        lines.append("### Матрица ошибок\n\n" + md_table + "\n")
        
        lines.append("\n### Качество распознавания по группам")
        for group, stats in group_report.items():
            lines.append(f"1. **{group}**")
            lines.append(f"   - Верно распознано: {stats['correct']} из {stats['total']} ({stats['accuracy_pct']}%)")
            if stats['common_errors']:
                error_strs = [f'"{err[0]}" ({err[1]})' for err in stats['common_errors']]
                lines.append(f"   - Типичные ошибки: путает с {', '.join(error_strs)}")
            else:
                lines.append("   - Типичные ошибки: отсутствуют")
            lines.append("")

        # Рекомендации
        lines.append("### Рекомендации")
        lines.append("Для улучшения точности:\n")
        lines.append("✅ Добавьте примеры для слабо представленных групп\n")
        lines.append("✅ Проверьте и уточните формулировки групп с частыми ошибками\n")
        lines.append("✅ Используйте множество обучающих примеров, оптимально от 10 тыс.\n")

        report_text = "\n".join(lines)

        return report_text
