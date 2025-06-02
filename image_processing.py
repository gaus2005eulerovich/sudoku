# Импорт необходимых библиотек
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from google.colab import files
from skimage.segmentation import clear_border

# 1. Загрузка предварительно обученной модели для распознавания цифр
model = load_model('/content/sudoku_model_v2.h5')

# 2. Класс для работы с судоку
class Sudoku:
    def __init__(self, board):
        """Инициализация доски 9x9"""
        self.board = board  # Доска представлена как двумерный список

    def show(self):
        """Отображение доски в удобочитаемом формате"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)  # Горизонтальные разделители
            row = []
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row.append("|")  # Вертикальные разделители
                row.append(str(self.board[i][j]) if self.board[i][j] != 0 else ".")
            print(" ".join(row))

    def solve(self):
        """Основной метод для решения судоку"""
        empty = self.find_empty()
        if not empty:
            return True  # Все клетки заполнены
        
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = 0  # Откат
        return False

    def find_empty(self):
        """Поиск первой пустой клетки (значение 0)"""
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(self, row, col, num):
        """Проверка допустимости числа в позиции (row, col)"""
        # Проверка строки
        if num in self.board[row]:
            return False
            
        # Проверка столбца
        for i in range(9):
            if self.board[i][col] == num:
                return False
                
        # Проверка блока 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.board[start_row + i][start_col + j] == num:
                    return False
        return True

# 3. Функции обработки изображения
def find_puzzle(image):
    """Поиск контура судоку на изображении"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # Бинаризация изображения
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Топ-5 контуров
    
    # Поиск четырехугольника (контура судоку)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
            
    raise Exception("Контур судоку не найден")

def extract_digit(cell):
    """Извлечение цифры из изображения ячейки"""
    # Бинаризация и очистка изображения
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = clear_border(thresh)
    
    # Поиск контуров цифры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Фильтрация мелких контуров
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Проверка заполненности области
    if cv2.countNonZero(mask) / (thresh.size) < 0.03:
        return None
        
    return cv2.bitwise_and(thresh, thresh, mask=mask)

def order_points(pts):
    """Упорядочивание точек контура"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Разница между координатами
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    
    return rect

def warp_image(image, rect):
    """Выравнивание изображения судоку"""
    (tl, tr, br, bl) = rect
    
    # Расчет размеров нового изображения
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))
    
    # Точки для преобразования
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Перспективное преобразование
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height)), M, max_width, max_height

# 4. Основной процесс
def main():
    # Шаг 1: Загрузка изображения
    uploaded = files.upload()
    if not uploaded:
        print("Ошибка: Изображение не загружено")
        return
        
    # Чтение и подготовка изображения
    image_data = next(iter(uploaded.values()))
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (600, 600))
    orig = image.copy()

    # Шаг 2: Поиск и выравнивание судоку
    try:
        puzzle_cnt = find_puzzle(image)
        rect = order_points(puzzle_cnt)
        warped, M, max_width, max_height = warp_image(orig, rect)
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return

    # Шаг 3: Разбиение на ячейки и распознавание цифр
    board = np.zeros((9, 9), dtype=int)
    step_x, step_y = warped.shape[1] // 9, warped.shape[0] // 9
    cell_locs = []

    for y in range(9):
        row = []
        for x in range(9):
            # Координаты ячейки
            start_x, end_x = x * step_x, (x + 1) * step_x
            start_y, end_y = y * step_y, (y + 1) * step_y
            row.append((start_x, start_y, end_x, end_y))
            
            # Обработка ячейки
            cell = warped[start_y:end_y, start_x:end_x]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            digit_img = extract_digit(gray)
            
            if digit_img is None:
                continue  # Пустая клетка
                
            # Подготовка изображения для нейросети
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = digit_img.astype("float") / 255.0
            digit_img = img_to_array(digit_img)
            digit_img = np.expand_dims(digit_img, axis=0)
            
            # Распознавание цифры
            predictions = model.predict(digit_img, verbose=0)[0]
            digit = np.argmax(predictions)
            confidence = predictions[digit]
            
            # Фильтр по уверенности
            if confidence > 0.8:
                board[y][x] = digit
                
        cell_locs.append(row)

    # Шаг 4: Решение судоку
    print("\nРаспознанная доска:")
    sudoku = Sudoku(board.tolist())
    sudoku.show()
    
    print("\nРешение...")
    if sudoku.solve():
        print("\nРешенная доска:")
        sudoku.show()
    else:
        print("\nРешение не найдено")
        return
        
    # Шаг 5: Визуализация результатов
    solved_image = warped.copy()
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:  # Только для пустых клеток
                start_x, start_y, end_x, end_y = cell_locs[y][x]
                # Центр ячейки
                center_x = int((start_x + end_x) / 2) - 10
                center_y = int((start_y + end_y) / 2) + 10
                # Наложение цифры
                cv2.putText(
                    solved_image, 
                    str(sudoku.board[y][x]), 
                    (center_x, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (0, 255, 0), 
                    3
                )
    
    # Обратное преобразование
    M_inv = cv2.getPerspectiveTransform(
        np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32"),
        rect
    )
    result = cv2.warpPerspective(solved_image, M_inv, (orig.shape[1], orig.shape[0]))
    
    # Сохранение и отображение
    cv2.imwrite("solved_sudoku.png", result)
    print("\nРезультат сохранен как 'solved_sudoku.png'")
    
    # Отображение в Colab
    from IPython.display import Image, display
    display(Image("solved_sudoku.png"))

# Запуск программы
if __name__ == "__main__":
    main()
